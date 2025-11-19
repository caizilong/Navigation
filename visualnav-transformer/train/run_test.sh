#!/bin/bash
# MambaViNT测试脚本 - 解决triton编译libcuda.so链接问题

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nomad_train

# 设置CUDA相关环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 关键: 确保gcc能找到64位libcuda.so
export LIBRARY_PATH=/lib/x86_64-linux-gnu:$LIBRARY_PATH

# 禁用i386库路径避免链接冲突
export CPATH=/usr/include/x86_64-linux-gnu:$CPATH

cd /home/czl/Navigation/visualnav-transformer/train

# 运行测试
python vint_train/models/mamba/test_mamba_vint.py
