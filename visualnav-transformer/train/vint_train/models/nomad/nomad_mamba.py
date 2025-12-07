"""
NoMaD-Mamba: 将NoMaD的Transformer替换为Mamba
保留所有其他组件（Goal Masking、Diffusion Policy等）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet

# 导入Mamba组件
from vint_train.models.mamba.mamba2 import Mamba2

# 使用官方 mamba_ssm.Block（与 MTIL 一致）
from mamba_ssm.modules.block import Block


class NoMaD_Mamba(nn.Module):
    """
    NoMaD with Mamba backbone

    关键改进：
    1. Transformer → Mamba2: 时序建模替换
    2. 保留Goal Masking机制（通过特殊处理实现）
    3. 保留GroupNorm优化
    4. 输出与NoMaD_ViNT完全兼容
    """

    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        # Mamba特定参数
        mamba_d_state: Optional[int] = 64,
        mamba_d_conv: Optional[int] = 4,
        mamba_expand: Optional[int] = 2,
        mamba_headdim: Optional[int] = 64,
        mamba_num_blocks: Optional[int] = 2,  # NoMaD原本用2层Transformer
        mamba_chunk_size: Optional[int] = 256,
        mamba_use_mem_eff: Optional[bool] = True,
    ) -> None:
        """
        NoMaD Mamba Encoder

        Args:
            context_size: 上下文帧数
            obs_encoder: 视觉编码器类型
            obs_encoding_size: 编码维度
            mamba_*: Mamba参数（与MambaViNT一致）
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # 1. 视觉编码器（与NoMaD_ViNT相同）
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(
                obs_encoder, in_channels=3)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError

        # 目标编码器（6通道：obs+goal）
        self.goal_encoder = EfficientNet.from_name(
            "efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

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

        def mlp_fn(dim):
            hidden_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

        self.mamba_blocks = nn.ModuleList([
            Block(
                dim=self.obs_encoding_size,
                mixer_cls=mixer_fn,
                mlp_cls=mlp_fn,
                norm_cls=nn.LayerNorm,
                fused_add_norm=True,  # 🚀 开启融合优化 (加速 20-30%)
                residual_in_fp32=True,  # 🚀 开启 FP32 残差 (提升稳定性)
            )
            for _ in range(mamba_num_blocks)
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
            (1 - goal_mask.float()) * ((self.context_size + 2) / (self.context_size + 1))
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

        # 1. 提取目标特征
        obsgoal_img = torch.cat(
            [obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)

        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(
                1)  # [batch, 1, encoding_size]
        goal_encoding = obsgoal_encoding

        # 2. 提取观测序列特征
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
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
        residual = None
        for block in self.mamba_blocks:
            obs_encoding, residual = block(obs_encoding, residual)

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


# ==================== 工具函数（与NoMaD_ViNT相同） ====================

def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


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
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module
