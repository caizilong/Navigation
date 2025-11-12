#!/bin/bash
# MTIL依赖安装脚本 - 在已有的nomad_train环境中添加Mamba相关依赖
# 使用方法: bash install_mamba_deps.sh

set -e  # 遇到错误立即退出

echo "======================================"
echo "开始为 nomad_train 环境添加 MTIL 依赖"
echo "======================================"

# 检查是否在正确的conda环境中
if [[ "$CONDA_DEFAULT_ENV" != "nomad_train" ]]; then
    echo "错误: 请先激活 nomad_train 环境"
    echo "运行: conda activate nomad_train"
    exit 1
fi

echo ""
echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 1. 安装 Mamba-SSM 及其依赖 (核心)
echo "======================================"
echo "步骤 1: 安装 Mamba-SSM 核心库"
echo "======================================"
echo "注意: 这需要 CUDA 和 C++ 编译器"
echo ""

# 先尝试安装 causal-conv1d (Mamba的依赖)
echo "安装 causal-conv1d..."
pip install causal-conv1d --no-build-isolation || {
    echo "警告: causal-conv1d 安装失败,尝试不使用 --no-build-isolation 标志..."
    pip install causal-conv1d
}

echo ""
echo "安装 mamba-ssm..."
pip install mamba-ssm --no-build-isolation || {
    echo "警告: mamba-ssm 安装失败,尝试不使用 --no-build-isolation 标志..."
    pip install mamba-ssm
}

# 2. 安装其他必需依赖
echo ""
echo "======================================"
echo "步骤 2: 安装其他必需依赖"
echo "======================================"

# einops (张量操作库,Mamba必需)
echo "安装 einops..."
pip install einops

# pytorch-lightning (训练框架)
echo "安装 pytorch-lightning..."
pip install pytorch-lightning

# huggingface_hub (Mamba模型加载)
echo "安装 huggingface_hub..."
pip install huggingface_hub

# 3. 可选的额外依赖 (如果需要MTIL的完整功能)
echo ""
echo "======================================"
echo "步骤 3: 安装可选依赖 (用于完整MTIL功能)"
echo "======================================"

read -p "是否安装 MuJoCo 相关依赖? (用于仿真环境) [y/N]: " install_mujoco
if [[ "$install_mujoco" == "y" || "$install_mujoco" == "Y" ]]; then
    echo "安装 mujoco 和 dm_control..."
    pip install mujoco==2.3.7 dm_control==1.0.14
    echo "MuJoCo 依赖安装完成"
else
    echo "跳过 MuJoCo 依赖"
fi

# 4. 验证安装
echo ""
echo "======================================"
echo "步骤 4: 验证安装"
echo "======================================"

python -c "
import sys
errors = []

# 检查核心依赖
try:
    import mamba_ssm
    print('✓ mamba-ssm 安装成功')
except ImportError as e:
    errors.append('✗ mamba-ssm 安装失败: ' + str(e))

try:
    import causal_conv1d
    print('✓ causal-conv1d 安装成功')
except ImportError as e:
    errors.append('✗ causal-conv1d 安装失败: ' + str(e))

try:
    from einops import rearrange, repeat
    print('✓ einops 安装成功')
except ImportError as e:
    errors.append('✗ einops 安装失败: ' + str(e))

try:
    import pytorch_lightning
    print('✓ pytorch-lightning 安装成功')
except ImportError as e:
    errors.append('✗ pytorch-lightning 安装失败: ' + str(e))

try:
    import huggingface_hub
    print('✓ huggingface_hub 安装成功')
except ImportError as e:
    errors.append('✗ huggingface_hub 安装失败: ' + str(e))

# 检查已有依赖
try:
    import torch
    print(f'✓ torch {torch.__version__} 已安装')
except ImportError:
    errors.append('✗ torch 未安装')

try:
    import torchvision
    print('✓ torchvision 已安装')
except ImportError:
    errors.append('✗ torchvision 未安装')

try:
    import efficientnet_pytorch
    print('✓ efficientnet_pytorch 已安装')
except ImportError:
    errors.append('✗ efficientnet_pytorch 未安装')

try:
    import diffusers
    print('✓ diffusers 已安装')
except ImportError:
    errors.append('✗ diffusers 未安装')

if errors:
    print('\n⚠️  发现以下错误:')
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print('\n✅ 所有依赖安装成功!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ MTIL 依赖安装完成!"
    echo "======================================"
    echo ""
    echo "环境已准备就绪,可以开始集成 Mamba 到 visualnav-transformer"
    echo ""
    echo "后续步骤:"
    echo "1. 可以开始将 MTIL 的 Mamba 模块移植到 ViNT/NoMaD"
    echo "2. 参考 MTIL/train/mamba_policy.py 中的实现"
    echo "3. 测试 Mamba2 模块是否能正常导入和运行"
else
    echo ""
    echo "======================================"
    echo "❌ 安装过程中出现错误"
    echo "======================================"
    echo ""
    echo "请检查错误信息并解决后重试"
    echo "常见问题:"
    echo "1. CUDA 版本不匹配 - 检查 cudatoolkit 版本"
    echo "2. 缺少 C++ 编译器 - 安装 gcc/g++"
    echo "3. 网络问题 - 配置国内镜像源"
    exit 1
fi
