# NoMaD-Mamba + GAIL 融合数据流图

## 一、整体架构数据流

```mermaid
graph TB
    subgraph "数据准备阶段"
        A[专家轨迹数据<br/>Expert Trajectories] --> B[数据加载器<br/>DataLoader]
        B --> C1[观测图像序列<br/>obs_img: B×3×context+1×H×W]
        B --> C2[目标图像<br/>goal_img: B×3×H×W]
        B --> C3[专家动作<br/>expert_action: B×8×2]
        B --> C4[距离标签<br/>dist_label: B]
    end

    subgraph "策略网络 - NoMaD-Mamba"
        D1[EfficientNet编码器<br/>Visual Encoder] --> E1[观测特征<br/>obs_encoding]
        D2[EfficientNet编码器<br/>Goal Encoder] --> E2[目标特征<br/>goal_encoding]
        E1 --> F[Mamba2时序建模<br/>Temporal Modeling]
        E2 --> F
        F --> G[全局特征<br/>encoding_size: 256]
        G --> H[Diffusion Policy<br/>U-Net + DDPM]
        H --> I[预测动作<br/>predicted_action: B×8×2]
    end

    subgraph "判别器网络 - Discriminator"
        J1[共享视觉编码器<br/>Shared EfficientNet] --> K1[状态编码<br/>state_encoding]
        J2[动作输入<br/>action: B×8×2] --> K2[动作展平<br/>flattened_action]
        K1 --> L[拼接<br/>Concat]
        K2 --> L
        L --> M[判别网络<br/>MLP + Spectral Norm]
        M --> N[真实性评分<br/>D_score: B×1]
    end

    subgraph "损失计算与优化"
        O1[BC Loss<br/>行为克隆损失] --> P[总损失<br/>Total Loss]
        O2[GAIL Loss<br/>对抗学习损失] --> P
        O3[Distance Loss<br/>距离预测损失] --> P
        P --> Q[梯度反向传播<br/>Backpropagation]
        Q --> R1[更新策略网络<br/>Update Policy]
        Q --> R2[更新判别器<br/>Update Discriminator]
    end

    C1 --> D1
    C2 --> D2
    C1 --> J1
    C2 --> J1
    I --> J2
    C3 --> J2
    
    I --> O1
    C3 --> O1
    N --> O2
    I --> O3
    C4 --> O3
```

## 二、训练阶段详细数据流

```mermaid
graph TB
    subgraph "Step 1: 策略网络前向传播"
        A1[Batch数据] --> B1{数据增强<br/>ColorJitter/Blur}
        B1 --> C1[obs_img: B×3×context+1×96×96]
        B1 --> C2[goal_img: B×3×96×96]
        
        C1 --> D1[分离上下文帧<br/>Split into context+1 frames]
        D1 --> E1[EfficientNet特征提取<br/>每帧独立编码]
        E1 --> F1[特征序列<br/>B×context+1×256]
        
        C2 --> D2[与当前观测拼接<br/>6通道输入]
        D2 --> E2[Goal Encoder<br/>EfficientNet]
        E2 --> F2[目标特征<br/>B×1×256]
        
        F1 --> G1[拼接<br/>Concat]
        F2 --> G1
        G1 --> H1[完整序列<br/>B×context+2×256]
        
        H1 --> I1{Goal Mask?<br/>prob=0.5}
        I1 -->|Mask| J1[置零最后一个token]
        I1 -->|No Mask| J2[保持原样]
        J1 --> K1[Mamba2 Block 1<br/>SSM + MLP + DropPath]
        J2 --> K1
        K1 --> K2[Mamba2 Block 2]
        K2 --> K3[Mamba2 Block 3]
        K3 --> K4[Mamba2 Block 4]
        K4 --> L1[平均池化<br/>Mean Pooling]
        L1 --> M1[全局特征<br/>B×256]
    end

    subgraph "Step 2: Diffusion Policy 采样"
        M1 --> N1[添加噪声<br/>Noise Scheduler]
        N1 --> O1[扩散迭代 t=10...1]
        O1 --> P1[U-Net去噪<br/>Down: 64→128→256<br/>Up: 256→128→64]
        P1 --> Q1[预测动作<br/>predicted_action: B×8×2]
    end

    subgraph "Step 3: 判别器训练"
        Q1 --> R1[策略数据<br/>policy_data]
        A1 --> R2[专家数据<br/>expert_data]
        
        R1 --> S1{停止梯度<br/>detach}
        S1 --> T1[策略obs编码]
        R2 --> T2[专家obs编码]
        
        T1 --> U1[判别器D]
        T2 --> U1
        
        U1 --> V1[专家评分<br/>expert_score]
        U1 --> V2[策略评分<br/>policy_score]
        
        V1 --> W1[BCE Loss<br/>target=1]
        V2 --> W2[BCE Loss<br/>target=0]
        
        W1 --> X1[判别器损失<br/>D_loss]
        W2 --> X1
        
        X1 --> Y1[梯度惩罚<br/>Gradient Penalty]
        Y1 --> Z1[更新判别器参数<br/>D_optimizer.step]
    end

    subgraph "Step 4: 策略网络优化"
        Q1 --> AA1[计算GAIL奖励<br/>r = -log1-sigmoid D]
        AA1 --> AB1[GAIL Loss<br/>-r.mean]
        
        Q1 --> AC1[BC Loss]
        A1 --> AC1
        AC1 --> AD1[Diffusion Loss<br/>MSE去噪误差]
        
        Q1 --> AE1[Distance Loss]
        A1 --> AE1
        AE1 --> AF1[MSE距离预测]
        
        AD1 --> AG1{Epoch < 10?<br/>预热阶段}
        AF1 --> AG1
        AB1 --> AG1
        
        AG1 -->|是| AH1[loss = BC_loss + Dist_loss]
        AG1 -->|否| AH2[loss = γ×BC + 1-γ×GAIL + Dist]
        
        AH1 --> AI1[反向传播]
        AH2 --> AI1
        AI1 --> AJ1[梯度裁剪<br/>max_norm=1.0]
        AJ1 --> AK1[更新策略参数<br/>optimizer.step]
        AK1 --> AL1[EMA更新<br/>ema_decay=0.999]
    end
```

## 三、推理阶段数据流

```mermaid
graph TB
    subgraph "机器人部署推理流程"
        A[ROS相机话题<br/>CompressedImage] --> B[图像预处理<br/>Resize 96×96]
        B --> C[历史帧缓存<br/>context_queue]
        C --> D[拼接上下文<br/>obs_img: 1×3×context+1×96×96]
        
        E[目标图像<br/>goal_img: 1×3×96×96] --> F[NoMaD-Mamba<br/>推理模式]
        D --> F
        
        F --> G[EfficientNet编码]
        G --> H[Mamba2前向]
        H --> I[Diffusion采样<br/>DDPM 10步]
        I --> J[动作轨迹<br/>1×8×2]
        
        J --> K[反归一化<br/>×ACTION_STATS.std + mean]
        K --> L[取首步动作<br/>action0: v, ω]
        
        L --> M[PD控制器<br/>pd_controller.py]
        M --> N[Twist消息<br/>linear.x, angular.z]
        N --> O[发布到ROS<br/>/cmd_vel]
        
        O --> P[机器人执行]
        P --> Q[更新观测]
        Q --> A
    end

    subgraph "循环控制"
        R[达到目标?] -->|否| A
        R -->|是| S[停止导航]
    end
    
    P --> R
```

## 四、关键数据维度说明

### 输入数据
| 数据类型 | 维度 | 说明 |
|---------|------|------|
| obs_img | `[B, 3×(context+1), 96, 96]` | 观测图像序列,context=5 |
| goal_img | `[B, 3, 96, 96]` | 目标图像 |
| expert_action | `[B, 8, 2]` | 专家动作轨迹 (v, ω) |
| dist_label | `[B]` | 与目标的距离标签 |

### 中间特征
| 特征类型 | 维度 | 说明 |
|---------|------|------|
| obs_encoding | `[B, context+1, 256]` | 观测特征序列 |
| goal_encoding | `[B, 1, 256]` | 目标特征 |
| mamba_output | `[B, context+2, 256]` | Mamba2输出 |
| global_feature | `[B, 256]` | 全局池化特征 |

### 输出数据
| 输出类型 | 维度 | 说明 |
|---------|------|------|
| predicted_action | `[B, 8, 2]` | 预测动作轨迹 |
| dist_pred | `[B]` | 预测距离 |
| D_score | `[B, 1]` | 判别器评分 (0-1) |

### 损失项
| 损失类型 | 权重 | 公式 |
|---------|------|------|
| BC Loss | `γ = 0.6` | `MSE(pred_action, expert_action)` |
| GAIL Loss | `1-γ = 0.4` | `-log(1 - sigmoid(D(s,a)))` |
| Distance Loss | `α = 1e-4` | `MSE(pred_dist, dist_label)` |
| Gradient Penalty | `λ = 10.0` | `(‖∇D‖₂ - 1)²` |

## 五、训练超参数配置

```yaml
# 策略网络
batch_size: 1024
learning_rate: 1e-4
optimizer: AdamW
weight_decay: 0.05
max_grad_norm: 1.0
ema_decay: 0.999

# Mamba参数
mamba_num_blocks: 4
mamba_d_state: 64
mamba_dropout: 0.1
mamba_drop_path: 0.1

# Diffusion参数
num_diffusion_iters: 10
noise_scheduler: DDPM

# GAIL参数
use_gail: True
gail_gamma: 0.6  # BC权重
discriminator_lr: 1e-4
discriminator_hidden_dim: 256
discriminator_update_freq: 1
gail_warm_up_epochs: 10
gradient_penalty_weight: 10.0

# 数据增强
goal_mask_prob: 0.5
color_jitter: 0.3
p_blur: 0.1
p_gray: 0.1
```

## 六、关键流程节点说明

### 1. Goal Masking机制
- **概率**: 50%的样本会mask掉goal token
- **目的**: 增强模型鲁棒性,学习无目标导航
- **实现**: 将goal特征置零 + 调整平均池化权重

### 2. Mamba2时序建模
- **优势**: O(N)复杂度,处理长序列高效
- **残差连接**: FP32精度 + Fused Add Norm加速
- **DropPath**: 随机深度正则化,防止过拟合

### 3. Diffusion Policy采样
- **迭代次数**: 10步 (平衡速度与质量)
- **调度器**: DDPM线性噪声调度
- **条件**: 全局特征作为条件输入U-Net

### 4. 判别器对抗训练
- **更新频率**: 每1步策略更新执行1次判别器更新
- **梯度惩罚**: WGAN-GP稳定训练
- **Spectral Norm**: 限制判别器Lipschitz常数

### 5. BC正则化策略
- **预热阶段**: 前10 epoch纯BC训练
- **混合阶段**: γ=0.6 BC + 0.4 GAIL
- **目的**: 保持训练稳定性,防止GAIL发散

## 七、数据流关键时间点

| 阶段 | 输入 | 处理 | 输出 | 耗时 |
|-----|------|------|------|------|
| 视觉编码 | 图像 | EfficientNet | 特征 | ~15ms |
| 时序建模 | 特征序列 | Mamba2×4 | 全局特征 | ~8ms |
| 扩散采样 | 特征+噪声 | U-Net×10 | 动作 | ~50ms |
| 判别器 | 状态+动作 | MLP | 评分 | ~3ms |
| **总推理** | - | - | - | **~76ms (13Hz)** |

---

**注**: 
- 所有时间为单卡V100估计值
- Batch size影响吞吐,不影响单样本延迟
- 部署时可使用EMA模型提升性能


```mermaid
graph TB
    subgraph "数据准备阶段"
        A[专家轨迹数据<br/>Expert Trajectories] --> B[数据加载器<br/>DataLoader]
        B --> C1[观测图像序列<br/>obs_img: B×3×context+1×H×W]
        B --> C2[目标图像<br/>goal_img: B×3×H×W]
        B --> C3[专家动作<br/>expert_action: B×8×2]
        B --> C4[距离标签<br/>dist_label: B]
    end

    subgraph "策略网络 - NoMaD-Mamba"
        D1[EfficientNet编码器<br/>Visual Encoder] --> E1[观测特征<br/>obs_encoding]
        D2[EfficientNet编码器<br/>Goal Encoder] --> E2[目标特征<br/>goal_encoding]
        E1 --> F[Mamba2时序建模<br/>Temporal Modeling]
        E2 --> F
        F --> G[全局特征<br/>encoding_size: 256]
        G --> H[Diffusion Policy<br/>U-Net + DDPM]
        H --> I[预测动作<br/>predicted_action: B×8×2]
    end

    subgraph "判别器网络 - Discriminator"
        J1[共享视觉编码器<br/>Shared EfficientNet] --> K1[状态编码<br/>state_encoding]
        J2[动作输入<br/>action: B×8×2] --> K2[动作展平<br/>flattened_action]
        K1 --> L[拼接<br/>Concat]
        K2 --> L
        L --> M[判别网络<br/>MLP + Spectral Norm]
        M --> N[真实性评分<br/>D_score: B×1]
    end

    subgraph "自适应α计算"
        N --> O[计算平均置信度 c_t]
        O --> P[计算自适应权重 α_t]
    end

    subgraph "损失计算与优化"
        Q1[BC Loss<br/>行为克隆损失] --> R1[加权BC Loss]
        Q2[GAIL Loss<br/>对抗学习损失] --> R2[加权GAIL Loss]
        P --> R1
        P --> R2
        R1 --> S[模仿学习总损失 L_IL]
        R2 --> S
        S --> T[总损失<br/>Total Loss = L_IL + L_Dist]
        U[Distance Loss<br/>距离预测损失] --> T
        T --> V[梯度反向传播<br/>Backpropagation]
        V --> W1[更新策略网络<br/>Update Policy]
        V --> W2[更新判别器<br/>Update Discriminator]
    end

    %% 数据流向
    C1 --> D1
    C2 --> D2
    C1 --> J1
    C2 --> J1
    I --> J2
    C3 --> J2
    
    I --> Q1
    C3 --> Q1
    N --> Q2
    I --> U
    C4 --> U

    classDef formula fill:#f9f,stroke:#333;
    class O,P formula;

    %% 图外公式说明（可选，用于文档）
    %% c_t = mean(D_score)
    %% α_t = σ(β * (c_t - 0.5))
```

```mermaid
graph TB
    subgraph "数据准备阶段"
        A[专家轨迹数据<br/>Expert Trajectories] --> B[数据加载器<br/>DataLoader]
        B --> C1[观测图像序列<br/>obs_img: B×3×context+1×H×W]
        B --> C2[目标图像<br/>goal_img: B×3×H×W]
        B --> C3[专家动作<br/>expert_action: B×8×2]
        B --> C4[距离标签<br/>dist_label: B]
    end

    subgraph "特征编码"
        C1 --> D1[EfficientNet编码器<br/>Visual Encoder]
        C2 --> D2[EfficientNet编码器<br/>Goal Encoder]
        D1 --> E1[观测特征 obs_encoding]
        D2 --> E2[目标特征 goal_encoding]
    end

    subgraph "Mamba2时序建模"
        E1 --> F[构建输入序列]
        E2 --> F
        F --> G[Mamba2核心处理]
        G --> H[当前隐状态 h_t]
    end

    subgraph "在线拓扑记忆机制"
        H --> I[计算状态变化率 Δh_t]
        I --> J{变化率超过阈值?}
        J -- 是 --> K[创建新记忆节点]
        J -- 否 --> L[维持现有拓扑图]
        K --> M[拓扑记忆库]
        L --> M
        E2 --> N[当前目标作为查询]
        N --> O[动态检索相似记忆]
        M --> O
        O --> P[检索到的记忆 h_topo]
    end

    subgraph "状态融合与策略"
        H --> Q[门控融合当前状态与记忆]
        P --> Q
        Q --> R[最终全局特征 h_final]
        R --> S[Diffusion Policy<br/>U-Net + DDPM]
        S --> T[预测动作<br/>predicted_action: B×8×2]
    end

    subgraph "判别器网络"
        C1 --> U1[共享视觉编码器]
        C2 --> U1
        T --> U2[动作输入]
        C3 --> U2
        U1 --> V1[状态编码]
        U2 --> V2[动作展平]
        V1 --> W[特征拼接]
        V2 --> W
        W --> X[判别网络<br/>MLP + Spectral Norm]
        X --> Y[真实性评分 D_score]
    end

    subgraph "自适应α计算"
        Y --> Z[计算平均置信度 c_t]
        Z --> AA[计算自适应权重 α_t]
    end

    subgraph "损失计算与优化"
        T --> AB1[BC Loss]
        C3 --> AB1
        Y --> AB2[GAIL Loss]
        T --> AF[Distance Loss]
        C4 --> AF
        AB1 --> AC1[加权BC Loss]
        AB2 --> AC2[加权GAIL Loss]
        AA --> AC1
        AA --> AC2
        AC1 --> AD[模仿学习总损失 L_IL]
        AC2 --> AD
        AD --> AE[总损失<br/>Total Loss]
        AF --> AE
        AE --> AG[梯度反向传播]
        AG --> AH1[更新策略网络]
        AG --> AH2[更新判别器]
    end

    classDef memory fill:#e6f7ff,stroke:#1890ff;
    classDef adaptive fill:#f6ffed,stroke:#52c41a;
    class M,O,P,Q,Z,AA,AC1,AC2,AD memory;
    class Z,AA,AC1,AC2,AD adaptive;
```