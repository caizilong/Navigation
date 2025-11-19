#!/bin/bash

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nomad_train

# 设置库路径,优先使用64位库
export LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 移除32位库路径
export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed 's|/lib/i386-linux-gnu:||g')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/lib/i386-linux-gnu:||g')

# 启动训练
cd /home/czl/Navigation/visualnav-transformer/train
python train.py -c config/vint_mamba.yaml
