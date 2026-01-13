#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mamba
export PYTHONPATH="/home/zhang1037/project/cai/Navigation/visualnav-mamba/diffusion_policy:$PYTHONPATH"

# 使用 nohup 后台运行，日志输出到 training.log
nohup python3 train.py -c ./config/defaults.yaml > training.log 2>&1 &
echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f training.log"
