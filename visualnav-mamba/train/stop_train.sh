#!/bin/bash

# 只查找当前用户在 visualnav-mamba 目录下运行的训练进程
CURRENT_USER=$(whoami)
PIDS=$(ps aux | grep "$CURRENT_USER" | grep "visualnav-mamba.*train.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有找到 $CURRENT_USER 的 visualnav-mamba 训练进程"
else
    echo "找到 $CURRENT_USER 的训练进程: $PIDS"
    echo "正在停止..."
    kill $PIDS
    sleep 2
    
    # 检查是否成功停止
    REMAINING=$(ps aux | grep "$CURRENT_USER" | grep "visualnav-mamba.*train.py" | grep -v grep | awk '{print $2}')
    if [ -z "$REMAINING" ]; then
        echo "训练进程已成功停止"
    else
        echo "进程仍在运行，尝试强制停止..."
        kill -9 $REMAINING
        echo "已强制停止"
    fi
fi
