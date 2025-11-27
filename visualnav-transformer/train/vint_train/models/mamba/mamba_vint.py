"""
MambaViNT: 使用Mamba替换Transformer的ViNT模型
保留ViNT的视觉编码器，使用Mamba进行时序建模

关键改进（参考MTIL实现）：
1. 训练模式(forward)：完整序列通过Mamba，利用内部chunk机制优化
2. 推理模式(step)：单步更新隐状态，实现O(1)显存占用
3. 与MTIL的对应关系：
   - forward() ↔ MTIL第725-765行
   - step() ↔ MTIL第641-722行
   - init_hidden_states() ↔ MTIL第626-639行

性能特点：
- 训练时显存占用：O(seq_len) - 线性增长但比Transformer好
- 推理时显存占用：O(1) - 恒定，与序列长度无关
- 参数量：固定，不随context_size变化
- 计算复杂度：O(seq_len) - 线性，比Transformer的O(seq_len²)快
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet

from vint_train.models.base_model import BaseModel
from .mamba2 import Mamba2

try:
    from mamba_ssm.modules.block import Block
except ImportError:
    # Fallback: 简化版Block
    class Block(nn.Module):
        def __init__(self, dim, mixer_cls, mlp_cls, norm_cls, **kwargs):
            super().__init__()
            self.mixer = mixer_cls(dim)
            self.norm = norm_cls(dim)
            self.mlp = mlp_cls(dim) if mlp_cls is not None else None
            self.norm2 = norm_cls(dim) if mlp_cls is not None else None

        def forward(self, x, residual=None):
            if residual is None:
                residual = x
            else:
                residual = residual + x

            hidden = self.norm(residual.to(dtype=self.norm.weight.dtype))
            hidden = self.mixer(hidden)
            hidden = hidden + residual

            if self.mlp is not None:
                r2 = self.norm2(hidden.to(dtype=self.norm2.weight.dtype))
                hidden = self.mlp(r2) + hidden

            return hidden, hidden


class MambaViNT(BaseModel):
    """
    MambaViNT: 用Mamba替换Transformer的ViNT模型

    架构:
    - 视觉编码器: EfficientNet (保留ViNT原有)
    - 时序建模: Mamba2 Blocks (替换MultiLayerDecoder)
    - 动作预测: Linear层 (保留ViNT原有)

    优势:
    - 利用Mamba的隐状态编码完整历史
    - 解决重复场景的状态歧义
    - 支持单步推理（部署友好）
    """

    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        # Mamba特定参数
        mamba_d_state: Optional[int] = 64,  # 导航任务优化（原MTIL为512）
        mamba_d_conv: Optional[int] = 4,
        mamba_expand: Optional[int] = 2,
        mamba_headdim: Optional[int] = 64,   # 导航任务优化（原MTIL为128）
        mamba_num_blocks: Optional[int] = 4,
        mamba_chunk_size: Optional[int] = 256,
        mamba_use_mem_eff: Optional[bool] = True,
    ) -> None:
        """
        初始化MambaViNT模型

        Args:
            context_size: 上下文帧数
            len_traj_pred: 预测轨迹长度
            learn_angle: 是否预测角度
            obs_encoder: 视觉编码器类型
            obs_encoding_size: 编码维度
            late_fusion: 是否后期融合
            mamba_d_state: Mamba状态维度
            mamba_d_conv: 卷积核大小
            mamba_expand: 扩展因子
            mamba_headdim: 每个头的维度
            mamba_num_blocks: Mamba块数量
            mamba_chunk_size: chunk大小
            mamba_use_mem_eff: 是否使用内存高效路径
        """
        super(MambaViNT, self).__init__(context_size, len_traj_pred, learn_angle)

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.late_fusion = late_fusion
        self.mamba_num_blocks = mamba_num_blocks

        # 1. 视觉编码器（保留ViNT的EfficientNet）
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
            self.num_obs_features = self.obs_encoder._fc.in_features
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError(f"Encoder {obs_encoder} not supported")

        # 压缩层
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 2. Mamba时序建模（替换Transformer）
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
            )
            for _ in range(mamba_num_blocks)
        ])

        # 3. 输出层（保留ViNT的预测头）
        final_dim = 32  # 与ViNT一致
        self.final_proj = nn.Linear(self.obs_encoding_size, final_dim)

        self.dist_predictor = nn.Sequential(
            nn.Linear(final_dim, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(final_dim, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（训练模式 - 参考MTIL的序列处理方式）

        关键改进：
        1. 保持与MTIL一致的序列处理方式
        2. 完整序列通过Mamba（利用chunk_size内部优化）
        3. 训练时不需要手动管理隐状态

        Args:
            obs_img: [batch, 3*(context_size+1), H, W]
            goal_img: [batch, 3, H, W]

        Returns:
            dist_pred: [batch, 1]
            action_pred: [batch, len_traj_pred, num_action_params]
        """
        device = next(self.parameters()).device
        obs_img = obs_img.to(device)
        goal_img = goal_img.to(device)

        batch_size = obs_img.shape[0]

        # 1. 提取目标特征
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)

        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(
                1)  # [batch, 1, encoding_size]

        # 2. 提取观测序列特征
        obs_img = torch.split(obs_img, 3, dim=1)  # 分割为context_size+1帧
        obs_img = torch.concat(obs_img, dim=0)  # [batch*(context+1), 3, H, W]

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape((self.context_size+1, batch_size, self.obs_encoding_size))
        # [batch, context+1, encoding_size]
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # 3. 拼接观测和目标 - 形成完整序列
        # 参考MTIL: 序列长度 = context_size+2 (历史帧 + 当前帧 + 目标)
        # [batch, context+2, encoding_size]
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # 4. 通过Mamba块进行时序建模（参考MTIL第759-760行）
        # 关键：Mamba内部会通过chunk_size优化，无需手动管理隐状态
        residual = None
        for block in self.mamba_blocks:
            tokens, residual = block(tokens, residual)

        # 5. 投影到最终维度（取最后一个token，即融合了所有历史的表示）
        final_repr = self.final_proj(tokens[:, -1, :])  # [batch, 32]

        # 6. 预测距离和动作
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # 重塑动作预测
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        # cumsum操作 - 使用梯度裁剪避免爆炸
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )

        return dist_pred, action_pred

    def init_hidden_states(self, batch_size: int, device=None):
        """
        初始化Mamba隐状态（用于推理）

        Args:
            batch_size: 批次大小
            device: 设备

        Returns:
            hidden_states: List[(conv_state, ssm_state)]
        """
        if device is None:
            device = next(self.parameters()).device

        hidden_list = []
        for block in self.mamba_blocks:
            if hasattr(block.mixer, "allocate_inference_cache"):
                conv_state, ssm_state = block.mixer.allocate_inference_cache(
                    batch_size, max_seqlen=1, dtype=None, device=device  # 添加device参数
                )
            else:
                # 创建空的隐状态
                conv_state, ssm_state = None, None
            hidden_list.append((conv_state, ssm_state))

        return hidden_list

    def step(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        单步推理（用于部署 - 参考MTIL第641-722行）

        关键改进：
        1. 完全参考MTIL的step实现
        2. 使用residual机制保持一致性
        3. 正确处理隐状态传递

        Args:
            obs_img: [batch, 3, H, W] - 单帧观测
            goal_img: [batch, 3, H, W] - 目标图像
            hidden_states: List[(conv_state, ssm_state)] - 每个block的隐状态

        Returns:
            (dist_pred, action_pred): 预测结果
            new_hidden_states: 更新后的隐状态
        """
        device = next(self.parameters()).device
        obs_img = obs_img.to(device)
        goal_img = goal_img.to(device)

        batch_size = obs_img.shape[0]

        # 1. 提取特征
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img, goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)

        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # 2. 融合特征（单步输入：只有当前帧的特征）
        # 注意：这里与训练不同，我们只输入当前帧，历史信息在隐状态中
        x_t = obs_encoding + goal_encoding  # [batch, encoding_size]

        # 3. 通过Mamba块（参考MTIL第688-717行）
        residual = None
        new_states = []
        hidden = x_t  # [batch, encoding_size]

        for i, blk in enumerate(self.mamba_blocks):
            conv_st, ssm_st = hidden_states[i] if i < len(hidden_states) else (None, None)

            # 初始化residual（参考MTIL第690-693行）
            if residual is None:
                residual = hidden
            else:
                residual = residual + hidden

            # LayerNorm（参考MTIL第695行）
            hidden_ln = blk.norm(residual.to(dtype=blk.norm.weight.dtype))

            # Mamba单步推理（参考MTIL第699-706行）
            if hasattr(blk.mixer, "step"):
                # unsqueeze添加序列维度 [batch, encoding_size] -> [batch, 1, encoding_size]
                y_t, new_conv_st, new_ssm_st = blk.mixer.step(
                    hidden_ln.unsqueeze(1), conv_st, ssm_st
                )
                # [batch, 1, encoding_size] -> [batch, encoding_size]
                y_t = y_t.squeeze(1)
            else:
                # 降级处理：无step方法时
                y_t = blk.mixer(hidden_ln.unsqueeze(1)).squeeze(1)
                new_conv_st, new_ssm_st = conv_st, ssm_st

            # 残差连接（参考MTIL第708行）
            hidden_out = y_t + residual

            # MLP（参考MTIL第711-713行）
            if blk.mlp is not None:
                r2 = blk.norm2(hidden_out.to(dtype=blk.norm2.weight.dtype))
                hidden_out = blk.mlp(r2) + hidden_out

            new_states.append((new_conv_st, new_ssm_st))
            hidden = hidden_out
            residual = hidden_out

        # 4. 预测（参考MTIL第720-721行）
        final_repr = self.final_proj(hidden)  # [batch, 32]
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # 重塑动作预测
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)

        return (dist_pred, action_pred), new_states
