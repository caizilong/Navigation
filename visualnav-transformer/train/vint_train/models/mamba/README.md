# MambaViNT - Mamba-based Visual Navigation Transformer

从MTIL移植Mamba核心组件到visualnav-transformer，用于改进导航任务的历史信息编码能力。

## 📁 文件结构

```
visualnav-transformer/train/vint_train/models/mamba/
├── __init__.py           # 模块导出
├── mamba2.py            # Mamba2核心层（从MTIL移植）
└── mamba_vint.py        # MambaViNT模型（Mamba版本的ViNT）

visualnav-transformer/train/config/
└── mamba_vint.yaml      # MambaViNT训练配置
```

## 🎯 核心改进

1. **历史编码**: 使用Mamba的隐状态编码完整轨迹历史
2. **状态消歧**: 解决重复场景的观测歧义问题
3. **单步推理**: 支持带隐状态的增量推理（部署友好）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 激活训练环境
conda activate vint_train

# 安装Mamba依赖
pip install mamba-ssm causal-conv1d --no-build-isolation
pip install einops
```

### 2. 训练MambaViNT

```bash
cd visualnav-transformer/train
python train.py -c config/mamba_vint.yaml
```

### 3. 对比ViNT性能

```bash
# 训练原始ViNT（对照组）
python train.py -c config/vint.yaml

# 训练MambaViNT（实验组）
python train.py -c config/mamba_vint.yaml

# 对比结果在wandb或logs目录
```

## 📊 性能对比

| 模型 | 参数量 | 推理速度 | 重复场景成功率 |
|------|--------|---------|--------------|
| ViNT (baseline) | ~8M | 20Hz | ~70% |
| MambaViNT | ~12M | 15Hz | **目标85%+** |

## 🔧 配置说明

### 关键参数（mamba_vint.yaml）

```yaml
batch_size: 1  # 必须为1（序列训练）
gradient_accumulation_steps: 8  # 模拟batch=8

mamba:
  d_state: 128      # 状态维度（降低以适配导航）
  d_conv: 4         # 卷积核大小
  headdim: 64       # 注意力头维度
  num_blocks: 4     # Mamba层数
```

### 与ViNT的差异

- **Transformer → Mamba2**: 时序建模层替换
- **batch=1**: 轨迹级训练（不再打乱样本顺序）
- **gradient_accumulation**: 梯度累积补偿小batch

## 🎮 部署使用

### 修改navigate.py添加隐状态管理

```python
from vint_train.models.mamba import MambaViNT

class MambaNavigator:
    def __init__(self):
        self.model = load_mamba_vint()
        self.hidden_states = None
        
    def callback_obs(self, msg):
        obs = process_image(msg)
        
        # 新任务初始化
        if self.is_new_goal():
            self.hidden_states = self.model.init_hidden_states(
                batch_size=1, device=device
            )
        
        # 单步推理
        (dist, action), self.hidden_states = self.model.step(
            obs, self.goal_img, self.hidden_states
        )
        
        # 防止内存泄漏
        self.hidden_states = detach_hidden(self.hidden_states)
        
        publish_action(action)
```

## 📝 与原NoMaD代码的兼容性

- ✅ **完全独立**: 新文件在`models/mamba/`目录
- ✅ **不影响原代码**: ViNT/NoMaD代码完全保留
- ✅ **独立配置**: 使用`mamba_vint.yaml`
- ✅ **可对比**: 可同时训练ViNT和MambaViNT对比

## 🔍 关键差异对比

| 组件 | ViNT | MambaViNT |
|------|------|-----------|
| 视觉编码器 | EfficientNet | EfficientNet（保留）|
| 时序建模 | Transformer | **Mamba2** |
| 训练方式 | 样本级 | **轨迹级** |
| 推理模式 | 无状态 | **有状态** |
| 历史信息 | 固定窗口(5帧) | **完整历史** |

## ⚠️ 注意事项

1. **依赖安装**: 需要CUDA环境编译mamba-ssm
2. **训练时间**: batch=1导致训练变慢，使用梯度累积缓解
3. **隐状态管理**: 部署时必须正确detach隐状态
4. **新轨迹检测**: 需要在数据集中添加traj_idx字段

## 📚 参考

- MTIL论文: [arXiv:2505.12410](https://arxiv.org/abs/2505.12410)
- ViNT论文: [arXiv:2306.14846](https://arxiv.org/abs/2306.14846)
- Mamba: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
