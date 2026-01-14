"""
NoMaD-Mamba: 将NoMaD的Transformer替换为Mamba
保留所有其他组件（Goal Masking、Diffusion Policy等）

支持通过 timm 库加载多种视觉编码器，可通过配置文件切换模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Callable
import timm

# 导入Mamba组件
from vint_train.models.mamba.mamba2 import Mamba2

# 使用官方 mamba_ssm.Block（与 MTIL 一致）
from mamba_ssm.modules.block import Block

# 从 timm 导入 DropPath
from timm.layers import DropPath


# ==================== 工具函数 ====================

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))

        # 获取源模块
        if isinstance(parent_module, nn.Sequential) and k.isdigit():
            # Sequential 且索引是数字
            src_module = parent_module[int(k)]
        else:
            # 普通属性访问
            src_module = getattr(parent_module, k)

        # 创建目标模块
        tgt_module = func(src_module)

        # 设置目标模块
        if isinstance(parent_module, nn.Sequential) and k.isdigit():
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.

    Args:
        root_module: 要处理的模块
        features_per_group: 每组的特征数，用于计算 num_groups

    Note:
        如果通道数无法被 features_per_group 整除，将动态计算合适的 num_groups
    """
    def get_num_groups(num_channels: int, features_per_group: int) -> int:
        """ 动态计算合适的 num_groups """
        if num_channels % features_per_group == 0:
            return num_channels // features_per_group
        # 尝试找到一个可以整除的 num_groups
        for divisor in [16, 8, 4, 2, 1]:
            if num_channels % divisor == 0:
                return divisor
        return 1

    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=get_num_groups(x.num_features, features_per_group),
            num_channels=x.num_features)
    )
    return root_module


# ==================== 视觉编码器工厂函数 ======================================

def create_vision_encoder(
    model_name: str = "efficientnet_b0",
    in_channels: int = 3,
    pretrained: bool = True,
    use_groupnorm: bool = True,
    features_per_group: int = 16,
) -> Tuple[nn.Module, int]:
    """
    使用 timm 创建视觉编码器

    Args:
        model_name: timm 支持的模型名称 (如 'efficientnet_b0', 'resnet50', 'convnext_tiny' 等)
        in_channels: 输入通道数
        pretrained: 是否使用预训练权重
        use_groupnorm: 是否将 BatchNorm 替换为 GroupNorm
        features_per_group: GroupNorm 每组的特征数

    Returns:
        model: 视觉编码器模型
        num_features: 输出特征维度
    """
    # 将配置中的 efficientnet-b0 格式转换为 timm 的 efficientnet_b0 格式
    model_name = model_name.replace("-", "_")

    # 创建没有分类头的特征提取器
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=0,  # 移除分类头，只保留特征提取部分
        global_pool='avg',  # 全局平均池化
    )

    # 替换 BatchNorm 为 GroupNorm（如果需要）
    # 注意：必须在测试 forward 之前进行替换，否则 MobileNetV4 等模型在小尺寸输入时
    # 会因为 BatchNorm 在 1x1 特征图上运行而报错
    if use_groupnorm:
        model = replace_bn_with_gn(model, features_per_group)

    # 通过实际 forward 获取正确的输出维度
    # 注意：model.num_features 在某些模型（如 MobileNetV4）上可能不准确
    # 使用 eval 模式确保即使有未替换的 BatchNorm 也不会报错
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 96, 96)
        dummy_output = model(dummy_input)
        num_features = dummy_output.shape[-1]
    model.train()  # 恢复训练模式

    return model, num_features


def get_supported_encoders() -> List[str]:
    """
    获取推荐使用的视觉编码器列表

    Returns:
        支持的编码器名称列表
    """
    return [
        # EfficientNet 系列
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5",
        "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l",
        # ResNet 系列
        "resnet18", "resnet34", "resnet50", "resnet101",
        # ConvNeXt 系列
        "convnext_tiny", "convnext_small", "convnext_base",
        # MobileNet V3 系列
        "mobilenetv3_small_100", "mobilenetv3_large_100",
        # MobileNetV4 系列 - 支持 96x96 输入（需要 use_groupnorm=True）
        "mobilenetv4_conv_small", "mobilenetv4_conv_small_035", "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_medium", "mobilenetv4_conv_large",
        "mobilenetv4_hybrid_medium", "mobilenetv4_hybrid_large",
        # DINOv3 系列 - 支持 96x96 和 160x160 输入
        "vit_small_patch16_dinov3", "vit_base_patch16_dinov3", "vit_large_patch16_dinov3",
        # DINOv2 系列 (注意：默认需要 224x224 或 518x518 输入)
        # "vit_small_patch14_dinov2", "vit_base_patch14_dinov2",  # 需要调整图像尺寸
        # 其他
        "regnetx_004", "regnety_004",
    ]


class NoMaD_Mamba(nn.Module):
    """
    NoMaD with Mamba backbone

    关键改进：
    1. Transformer → Mamba2: 时序建模替换
    2. 保留Goal Masking机制（通过特殊处理实现）
    3. 保留GroupNorm优化
    4. 输出与NoMaD_ViNT完全兼容
    5. 支持通过 timm 库配置不同的视觉编码器
    """

    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet_b0",
        # 新增：目标编码器配置，默认使用与 obs_encoder 相同的模型
        goal_encoder: Optional[str] = None,
        obs_encoding_size: Optional[int] = 512,
        pretrained: Optional[bool] = True,  # 新增：是否使用预训练权重
        # Mamba特定参数
        mamba_d_state: Optional[int] = 64,
        mamba_d_conv: Optional[int] = 4,
        mamba_expand: Optional[int] = 2,
        mamba_headdim: Optional[int] = 64,
        mamba_num_blocks: Optional[int] = 2,  # NoMaD原本用2层Transformer
        mamba_chunk_size: Optional[int] = 256,
        mamba_use_mem_eff: Optional[bool] = True,
        # [新增] 正则化参数
        mamba_dropout: Optional[float] = 0.0,      # Dropout比例
        mamba_drop_path: Optional[float] = 0.0,    # DropPath比例
    ) -> None:
        """
        NoMaD Mamba Encoder

        Args:
            context_size: 上下文帧数
            obs_encoder: 观测视觉编码器类型 (timm 支持的任意模型名称)
            goal_encoder: 目标视觉编码器类型，默认与 obs_encoder 相同
            obs_encoding_size: 编码维度
            pretrained: 是否使用预训练权重
            mamba_*: Mamba参数（与MambaViNT一致）
            mamba_dropout: Dropout比例
            mamba_drop_path: DropPath比例（Stochastic Depth）
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # 如果未指定 goal_encoder，使用与 obs_encoder 相同的模型
        if goal_encoder is None:
            goal_encoder = obs_encoder

        # 1. 视觉编码器 - 使用 timm 库创建
        # 观测编码器（3通道输入）
        self.obs_encoder, self.num_obs_features = create_vision_encoder(
            model_name=obs_encoder,
            in_channels=3,
            pretrained=pretrained,
            use_groupnorm=True,
        )

        # 目标编码器（6通道：obs+goal 拼接）
        self.goal_encoder, self.num_goal_features = create_vision_encoder(
            model_name=goal_encoder,
            in_channels=6,
            pretrained=False,  # 6通道无法使用预训练权重
            use_groupnorm=True,
        )

        # 压缩层
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 2. Mamba时序建模（替换TransformerEncoder）
        # [修改] 添加 Dropout 支持
        def mixer_fn(dim):
            return Mamba2(
                d_model=dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                headdim=mamba_headdim,
                chunk_size=mamba_chunk_size,
                use_mem_eff_path=mamba_use_mem_eff,
            )

        # [修改] MLP 中添加 Dropout
        def mlp_fn(dim):
            hidden_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(
                    mamba_dropout) if mamba_dropout > 0 else nn.Identity(),  # [新增]
                nn.Linear(hidden_dim, dim),
                nn.Dropout(
                    mamba_dropout) if mamba_dropout > 0 else nn.Identity(),  # [新增]
            )

        # [修改] 为每个 Block 添加 DropPath
        # 使用 stochastic depth: 每一层的 drop_path 概率递增
        dpr = [x.item() for x in torch.linspace(0, mamba_drop_path,
                                                mamba_num_blocks)]  # stochastic depth decay rule

        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                'block': Block(
                    dim=self.obs_encoding_size,
                    mixer_cls=mixer_fn,
                    mlp_cls=mlp_fn,
                    norm_cls=nn.LayerNorm,
                    fused_add_norm=True,  # 🚀 开启融合优化 (加速 20-30%)
                    residual_in_fp32=True,  # 🚀 开启 FP32 残差 (提升稳定性)
                ),
                # [新增]
                'drop_path': DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            })
            for i in range(mamba_num_blocks)
        ])

        # # 3. Goal Mask定义（保留NoMaD的mask机制）
        # self.goal_mask = torch.zeros(
        #     (1, self.context_size + 2), dtype=torch.bool)
        # self.goal_mask[:, -1] = True  # Mask out the goal
        # self.no_mask = torch.zeros(
        #     (1, self.context_size + 2), dtype=torch.bool)
        # self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        # self.avg_pool_mask = torch.cat([
        #     1 - self.no_mask.float(),
        #     (1 - self.goal_mask.float()) *
        #     ((self.context_size + 2) / (self.context_size + 1))
        # ], dim=0)

        # 3. Goal Mask定义（保留NoMaD的mask机制）
        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True  # Mask out the goal
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)

        # 注册为 buffer，这样 model.to(device) 时会自动移动
        self.register_buffer("goal_mask", goal_mask, persistent=True)
        self.register_buffer("no_mask", no_mask, persistent=True)

        # all_masks 与 avg_pool_mask 也注册为 buffer
        all_masks = torch.cat([no_mask, goal_mask], dim=0)
        self.register_buffer("all_masks", all_masks, persistent=True)

        avg_pool_mask = torch.cat([
            1 - no_mask.float(),
            (1 - goal_mask.float()) *
            ((self.context_size + 2) / (self.context_size + 1))
        ], dim=0)
        self.register_buffer("avg_pool_mask", avg_pool_mask, persistent=True)

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播（与NoMaD_ViNT接口兼容）

        Args:
            obs_img: [batch, 3*(context_size+1), H, W]
            goal_img: [batch, 3, H, W]
            input_goal_mask: [batch] - 1表示mask goal, 0表示不mask

        Returns:
            obs_encoding_tokens: [batch, obs_encoding_size] - 全局特征
        """
        device = obs_img.device
        batch_size = obs_img.size(0)

        # 1. 提取目标特征 (使用 timm 模型的 forward 方法，已包含全局池化)
        obsgoal_img = torch.cat(
            [obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        # timm 模型直接返回池化后的特征向量
        obsgoal_encoding = self.goal_encoder(obsgoal_img)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(
                1)  # [batch, 1, encoding_size]
        goal_encoding = obsgoal_encoding

        # 2. 提取观测序列特征
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        # timm 模型直接返回池化后的特征向量
        obs_encoding = self.obs_encoder(obs_img)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape(
            (self.context_size+1, batch_size, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # [batch, context+2, encoding_size]
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)

        # 3. 处理Goal Mask（Mamba版本的实现）
        # 注意：Mamba不像Transformer有内置的src_key_padding_mask
        # 我们通过将masked token的特征置零来实现类似效果
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
            # 对于mask=1的样本，将goal token（最后一个）的特征置零
            mask_indices = (goal_mask == 1).nonzero(as_tuple=True)[0]
            if len(mask_indices) > 0:
                obs_encoding[mask_indices, -1, :] = 0.0

        # 4. 通过Mamba块处理（替代Transformer）
        # [修改] 应用 DropPath
        residual = None
        for mamba_dict in self.mamba_blocks:
            block = mamba_dict['block']
            drop_path = mamba_dict['drop_path']

            # Mamba block forward
            obs_encoding, residual = block(obs_encoding, residual)

            # [新增] 应用 DropPath
            obs_encoding = drop_path(obs_encoding)

        # 5. 应用平均池化mask（与NoMaD_ViNT一致）
        if input_goal_mask is not None:
            no_goal_mask = input_goal_mask.long()
            avg_mask = torch.index_select(self.avg_pool_mask.to(
                device), 0, no_goal_mask).unsqueeze(-1)
            obs_encoding = obs_encoding * avg_mask

        # 6. 全局平均池化
        obs_encoding_tokens = torch.mean(
            obs_encoding, dim=1)  # [batch, encoding_size]

        return obs_encoding_tokens

    def __call__(self, *args, **kwargs):
        """
        重写 __call__ 方法以支持两种调用方式：
        1. 普通前向传播: model(obs_img, goal_img, input_goal_mask=None)
        2. 函数名调用: model('vision_encoder', obs_img=..., goal_img=..., input_goal_mask=...)
        """
        if len(args) == 1 and isinstance(args[0], str) and args[0] in ['vision_encoder', 'noise_pred_net', 'dist_pred_net']:
            # 函数名调用模式
            func_name = args[0]
            if func_name == "vision_encoder":
                return self(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
            else:
                raise NotImplementedError(
                    f"Function {func_name} is not implemented for NoMaD_Mamba")
        else:
            # 普通前向传播模式
            return super().__call__(*args, **kwargs)


# ==================== 向后兼容函数 ====================

__all__ = [
    'NoMaD_Mamba',
    'replace_bn_with_gn',
    'replace_submodules',
    'create_vision_encoder',
    'get_supported_encoders',
]
