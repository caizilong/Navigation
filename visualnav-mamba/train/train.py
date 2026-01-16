# ============================================================
# 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
# 这三个模块不依赖 CUDA，可以安全导入
# ============================================================

import os
import argparse
import yaml


def _early_set_gpu():
    """在 torch 初始化前设置 GPU"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", "-c", default="config/defaults.yaml", type=str)
    args, _ = parser.parse_known_args()

    with open("config/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    if "gpu_ids" in config:
        gpu_ids = config["gpu_ids"]
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
        print(
            f"[Early Init] Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


# 立即调用，确保在 import torch 之前执行
_early_set_gpu()

# ============================================================
# 现在可以安全地导入 torch 和其他依赖 CUDA 的模块
# ============================================================
from vint_train.training.train_eval_loop import (
    train_eval_loop_nomad,
    load_model,
)
from vint_train.data.data_utils import InterleavedSampler
from vint_train.data.vint_dataset import ViNT_Dataset
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad_mamba import NoMaD_Mamba, replace_bn_with_gn
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from warmup_scheduler import GradualWarmupScheduler
import copy
import pdb
import time
import numpy as np
import wandb
from torchvision import transforms
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, ConcatDataset
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch

# IMPORT YOUR MODEL HERE - 这些导入会触发 mamba_ssm 的导入


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    # 确保 gpu_ids 格式正确
    if "gpu_ids" not in config:
        config["gpu_ids"] = [0]
    elif isinstance(config["gpu_ids"], int):
        config["gpu_ids"] = [config["gpu_ids"]]

    # GPU 已在文件开头通过 _early_set_gpu() 设置
    if torch.cuda.is_available():
        print("Using cuda devices:", os.environ.get(
            "CUDA_VISIBLE_DEVICES", "all"))
    else:
        print("Using cpu")

    # 注意：设置 CUDA_VISIBLE_DEVICES 后，GPU 索引会重新映射
    # 例如：gpu_ids=[1] 设置 CUDA_VISIBLE_DEVICES=1 后，该 GPU 变成 cuda:0
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                # [新增] 仅在训练集上应用增强
                augmentations = config.get(
                    "augmentations", None) if data_split_type == "train" else None

                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                    augmentations=augmentations,  # [新增] 传入增强参数
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    # 读取配置中的优化参数
    pin_memory = config.get("pin_memory", True)
    prefetch_factor = config.get("prefetch_factor", 2)
    use_interleaved_sampler = config.get("use_interleaved_sampler", False)
    sampler_chunk_size = config.get(
        "sampler_chunk_size", config["batch_size"] * 4)

    # 多数据集时启用交错采样器（如果配置开启）
    is_multi_dataset = len(config["datasets"]) > 1

    if use_interleaved_sampler and is_multi_dataset:
        # 多数据集：使用交错采样器，减少 LMDB 缓存切换频率
        sampler = InterleavedSampler(
            train_dataset,
            chunk_size=sampler_chunk_size,
            shuffle=True
        )
        print(
            f"Using InterleavedSampler for {len(config['datasets'])} datasets with chunk_size={sampler_chunk_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
    else:
        # 单数据集或禁用交错采样：使用标准随机采样
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model (only nomad_mamba supported)
    if config["model_type"] != "nomad":
        raise ValueError(
            f"Only 'nomad' model_type is supported. Got: {config['model_type']}")

    if config["vision_encoder"] != "nomad_mamba":
        raise ValueError(
            f"Only 'nomad_mamba' vision_encoder is supported. Got: {config['vision_encoder']}")

    # NoMaD-Mamba vision encoder
    vision_encoder = NoMaD_Mamba(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        obs_encoder=config.get("obs_encoder", "efficientnet_b0"),
        goal_encoder=config.get("goal_encoder", None),
        pretrained=config.get("pretrained", True),
        mamba_d_state=config.get("mamba_d_state", 64),
        mamba_d_conv=config.get("mamba_d_conv", 4),
        mamba_expand=config.get("mamba_expand", 2),
        mamba_headdim=config.get("mamba_headdim", 64),
        mamba_num_blocks=config.get("mamba_num_blocks", 2),
        mamba_chunk_size=config.get("mamba_chunk_size", 256),
        mamba_use_mem_eff=config.get("mamba_use_mem_eff", True),
        mamba_dropout=config.get("mamba_dropout", 0.0),
        mamba_drop_path=config.get("mamba_drop_path", 0.0),
    )

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    # ========== 根据配置决定是否创建 EMA 模型 ==========
    use_ema = config.get("use_ema", False)
    ema_model = None

    if use_ema:
        # 由于 Mamba 模型包含无法 pickle 的 generator 对象，
        # 我们需要创建一个完全独立的模型实例作为 EMA 模型
        ema_vision_encoder = NoMaD_Mamba(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            obs_encoder=config.get("obs_encoder", "efficientnet_b0"),
            goal_encoder=config.get("goal_encoder", None),
            pretrained=config.get("pretrained", True),
            mamba_d_state=config.get("mamba_d_state", 64),
            mamba_d_conv=config.get("mamba_d_conv", 4),
            mamba_expand=config.get("mamba_expand", 2),
            mamba_headdim=config.get("mamba_headdim", 64),
            mamba_num_blocks=config.get("mamba_num_blocks", 2),
            mamba_chunk_size=config.get("mamba_chunk_size", 256),
            mamba_use_mem_eff=config.get("mamba_use_mem_eff", True),
            mamba_dropout=config.get("mamba_dropout", 0.0),
            mamba_drop_path=config.get("mamba_drop_path", 0.0),
        )

        ema_noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        ema_dist_pred_network = DenseNetwork(
            embedding_dim=config["encoding_size"])

        ema_model_instance = NoMaD(
            vision_encoder=ema_vision_encoder,
            noise_pred_net=ema_noise_pred_net,
            dist_pred_net=ema_dist_pred_network,
        )
        # 加载主模型的初始权重到 EMA 模型
        ema_model_instance.load_state_dict(model.state_dict())

        # 创建 EMAModel 包装器，使用配置中的 ema_decay 参数
        ema_decay = config.get("ema_decay", 0.9999)
        ema_model = EMAModel(
            model=ema_model_instance,
            power=0.75,
            update_after_step=0,
            inv_gamma=1.0,
            max_value=ema_decay,
        )
        print(f"Created independent EMA model instance with decay={ema_decay}")
    else:
        print("EMA model disabled (use_ema=False)")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # if config["clipping"]:
    #     print("Clipping gradients to", config["max_norm"])
    #     for p in model.parameters():
    #         if not p.requires_grad:
    #             continue
    #         p.register_hook(
    #             lambda grad: torch.clamp(
    #                 grad, -1 * config["max_norm"], config["max_norm"]
    #             )
    #         )
    if config.get("clipping", False):
        print("Will apply global gradient clipping with max_norm =",
              config.get("max_norm", 1.0))

    # --- 添加参数分组函数 ---
    def build_optimizer(model, config):
        """
        构建优化器, 将 weight_decay 应用于权重矩阵, 但跳过 Bias 和 Normalization 层
        """
        lr = float(config["lr"])
        weight_decay = config.get("weight_decay", 0.0)

        if weight_decay == 0.0:
            # 如果没有设置 weight_decay, 使用默认优化器
            optimizer_type = config["optimizer"].lower()
            if optimizer_type == "adam":
                return Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
            elif optimizer_type == "adamw":
                return AdamW(model.parameters(), lr=lr)
            elif optimizer_type == "sgd":
                return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                raise ValueError(f"Optimizer {optimizer_type} not supported")

        # 分组参数: 让 bias 和 layernorm 不参与 weight_decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)

        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # 所有 bias 都不衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 权重矩阵参与衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Norm层的权重不衰减
                    no_decay.add(fpn)
                elif "A_log" in pn or "D" in pn:
                    # Mamba 特有的参数 (SSM parameters), 通常不建议衰减
                    no_decay.add(fpn)

        # 验证参数覆盖情况
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )

        # 创建优化器参数组
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        print(
            f"Optimizer: {len(decay)} params with weight_decay={weight_decay}, {len(no_decay)} params with weight_decay=0.0")

        optimizer_type = config["optimizer"].lower()
        if optimizer_type == "adam":
            return Adam(optim_groups, lr=lr, betas=(0.9, 0.98))
        elif optimizer_type == "adamw":
            return AdamW(optim_groups, lr=lr)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(optim_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        # f"cuda:{}" if torch.cuda.is_available() else "cpu")
        latest_checkpoint = torch.load(latest_path)
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    # 注意：设置 CUDA_VISIBLE_DEVICES 后，设备 ID 会重新映射为 0, 1, 2...
    if len(config["gpu_ids"]) > 1:
        # 重新映射的设备 ID 列表
        remapped_device_ids = list(range(len(config["gpu_ids"])))
        model = nn.DataParallel(model, device_ids=remapped_device_ids)
    model = model.to(device)

    # EMA 模型也需要移到设备（如果启用）
    if ema_model is not None:
        ema_model.averaged_model = ema_model.averaged_model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(
                latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(
                latest_checkpoint["scheduler"].state_dict())

    # Train with NoMaD (diffusion policy)
    train_eval_loop_nomad(
        train_model=config["train"],
        model=model,
        ema_model=ema_model,  # 传入预先创建的 EMA 模型
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
        eval_print_log_freq=config.get(
            "eval_print_log_freq", config["print_log_freq"]),
        eval_wandb_log_freq=config.get(
            "eval_wandb_log_freq", config["wandb_log_freq"]),
        eval_image_log_freq=config.get(
            "eval_image_log_freq", config["image_log_freq"]),
    )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/mamba.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(),
            entity="coisinic243-beijing-university-of-technology",  # 使用你的wandb账户
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config.get(
            "run_name", f"{config['project_name']}_{int(time.time())}")
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
