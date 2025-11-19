import re
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def parse_complete_log(log_file_path):
    """
    更完整的日志解析器，专门针对你的日志格式
    """
    train_losses = {'dist_loss': [], 'action_loss': [], 'total_loss': []}
    train_sims = {'action_waypts_cos_sim': [], 'multi_action_waypts_cos_sim': [], 
                  'action_orien_cos_sim': [], 'multi_action_orien_cos_sim': []}
    test_losses = {'dist_loss': [], 'action_loss': [], 'total_loss': []}
    test_sims = {'action_waypts_cos_sim': [], 'multi_action_waypts_cos_sim': [], 
                 'action_orien_cos_sim': [], 'multi_action_orien_cos_sim': []}
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取训练指标（使用平均值）
    for metric in ['dist_loss', 'action_loss', 'total_loss', 'action_waypts_cos_sim', 
                   'multi_action_waypts_cos_sim', 'action_orien_cos_sim', 'multi_action_orien_cos_sim']:
        # 匹配训练平均值 - 从batch日志中提取
        pattern = rf'\(epoch (\d+)\) \(batch \d+/\d+\) {metric} \(train\): [\d.-]+ \(100pt moving_avg: [\d.-]+\) \(avg: ([\d.-]+)\)'
        matches = re.findall(pattern, content)
        values = [(int(epoch), float(value)) for epoch, value in matches]
        
        # 按epoch排序并提取值，去重（取每个epoch的最后一个值）
        epoch_dict = {}
        for epoch, value in values:
            epoch_dict[epoch] = value  # 后面的值会覆盖前面的值，实现"取最后一个"
        
        # 按顺序排列
        sorted_epochs = sorted(epoch_dict.keys())
        epoch_values = [epoch_dict[epoch] for epoch in sorted_epochs]
        
        if metric in train_losses:
            train_losses[metric] = epoch_values
        elif metric in train_sims:
            train_sims[metric] = epoch_values
    
    # 提取测试指标
    for metric in ['dist_loss', 'action_loss', 'total_loss', 'action_waypts_cos_sim', 
                   'multi_action_waypts_cos_sim', 'action_orien_cos_sim', 'multi_action_orien_cos_sim']:
        pattern = rf'\(epoch (\d+)\) {metric} \(go_stanford_test\) ([\d.-]+)'
        matches = re.findall(pattern, content)
        values = [(int(epoch), float(value)) for epoch, value in matches]
        
        # 按epoch排序并提取值，去重
        epoch_dict = {}
        for epoch, value in values:
            epoch_dict[epoch] = value  # 取最后一个值
        
        # 按顺序排列
        sorted_epochs = sorted(epoch_dict.keys())
        epoch_values = [epoch_dict[epoch] for epoch in sorted_epochs]
        
        if metric in test_losses:
            test_losses[metric] = epoch_values
        elif metric in test_sims:
            test_sims[metric] = epoch_values
    
    return train_losses, train_sims, test_losses, test_sims

def plot_training_curves(train_losses, train_sims, test_losses, test_sims, output_dir='training_plots'):
    """
    绘制训练曲线（使用Agg后端，不显示图形）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取最大的epoch数
    max_train_epochs = max([len(v) for v in train_losses.values() if v] + [len(v) for v in train_sims.values() if v] + [0])
    max_test_epochs = max([len(v) for v in test_losses.values() if v] + [len(v) for v in test_sims.values() if v] + [0])
    
    epochs_train = list(range(max_train_epochs)) if max_train_epochs > 0 else []
    epochs_test = list(range(max_test_epochs)) if max_test_epochs > 0 else []
    
    # 1. 损失函数图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    if epochs_train and 'dist_loss' in train_losses and len(train_losses['dist_loss']) > 0:
        plt.plot(epochs_train, train_losses['dist_loss'], label='Train Dist Loss', alpha=0.7)
    if epochs_test and 'dist_loss' in test_losses and len(test_losses['dist_loss']) > 0:
        plt.plot(epochs_test, test_losses['dist_loss'], label='Test Dist Loss', alpha=0.7)
    plt.title('Distance Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    if epochs_train and 'action_loss' in train_losses and len(train_losses['action_loss']) > 0:
        plt.plot(epochs_train, train_losses['action_loss'], label='Train Action Loss', alpha=0.7)
    if epochs_test and 'action_loss' in test_losses and len(test_losses['action_loss']) > 0:
        plt.plot(epochs_test, test_losses['action_loss'], label='Test Action Loss', alpha=0.7)
    plt.title('Action Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if epochs_train and 'total_loss' in train_losses and len(train_losses['total_loss']) > 0:
        plt.plot(epochs_train, train_losses['total_loss'], label='Train Total Loss', alpha=0.7)
    if epochs_test and 'total_loss' in test_losses and len(test_losses['total_loss']) > 0:
        plt.plot(epochs_test, test_losses['total_loss'], label='Test Total Loss', alpha=0.7)
    plt.title('Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 相似度指标图
    plt.subplot(2, 2, 4)
    if epochs_train and 'action_waypts_cos_sim' in train_sims and len(train_sims['action_waypts_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['action_waypts_cos_sim'], label='Train Waypoints Sim', alpha=0.7)
    if epochs_test and 'action_waypts_cos_sim' in test_sims and len(test_sims['action_waypts_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['action_waypts_cos_sim'], label='Test Waypoints Sim', alpha=0.7)
    plt.title('Action Waypoints Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 方向相似度图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if epochs_train and 'action_orien_cos_sim' in train_sims and len(train_sims['action_orien_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['action_orien_cos_sim'], label='Train Action Orientation', alpha=0.7)
    if epochs_test and 'action_orien_cos_sim' in test_sims and len(test_sims['action_orien_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['action_orien_cos_sim'], label='Test Action Orientation', alpha=0.7)
    plt.title('Action Orientation Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    if epochs_train and 'multi_action_waypts_cos_sim' in train_sims and len(train_sims['multi_action_waypts_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['multi_action_waypts_cos_sim'], label='Train Multi Waypoints', alpha=0.7)
    if epochs_test and 'multi_action_waypts_cos_sim' in test_sims and len(test_sims['multi_action_waypts_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['multi_action_waypts_cos_sim'], label='Test Multi Waypoints', alpha=0.7)
    plt.title('Multi Action Waypoints Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if epochs_train and 'multi_action_orien_cos_sim' in train_sims and len(train_sims['multi_action_orien_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['multi_action_orien_cos_sim'], label='Train Multi Orientation', alpha=0.7)
    if epochs_test and 'multi_action_orien_cos_sim' in test_sims and len(test_sims['multi_action_orien_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['multi_action_orien_cos_sim'], label='Test Multi Orientation', alpha=0.7)
    plt.title('Multi Action Orientation Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 综合性能图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    if epochs_train and 'dist_loss' in train_losses and len(train_losses['dist_loss']) > 0:
        plt.plot(epochs_train, train_losses['dist_loss'], label='Train', alpha=0.7)
    if epochs_test and 'dist_loss' in test_losses and len(test_losses['dist_loss']) > 0:
        plt.plot(epochs_test, test_losses['dist_loss'], label='Test', alpha=0.7)
    plt.title('Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    if epochs_train and 'action_loss' in train_losses and len(train_losses['action_loss']) > 0:
        plt.plot(epochs_train, train_losses['action_loss'], label='Train', alpha=0.7)
    if epochs_test and 'action_loss' in test_losses and len(test_losses['action_loss']) > 0:
        plt.plot(epochs_test, test_losses['action_loss'], label='Test', alpha=0.7)
    plt.title('Action Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    if epochs_train and 'total_loss' in train_losses and len(train_losses['total_loss']) > 0:
        plt.plot(epochs_train, train_losses['total_loss'], label='Train', alpha=0.7)
    if epochs_test and 'total_loss' in test_losses and len(test_losses['total_loss']) > 0:
        plt.plot(epochs_test, test_losses['total_loss'], label='Test', alpha=0.7)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    if epochs_train and 'action_waypts_cos_sim' in train_sims and len(train_sims['action_waypts_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['action_waypts_cos_sim'], label='Train', alpha=0.7)
    if epochs_test and 'action_waypts_cos_sim' in test_sims and len(test_sims['action_waypts_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['action_waypts_cos_sim'], label='Test', alpha=0.7)
    plt.title('Waypoints Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    if epochs_train and 'action_orien_cos_sim' in train_sims and len(train_sims['action_orien_cos_sim']) > 0:
        plt.plot(epochs_train, train_sims['action_orien_cos_sim'], label='Train', alpha=0.7)
    if epochs_test and 'action_orien_cos_sim' in test_sims and len(test_sims['action_orien_cos_sim']) > 0:
        plt.plot(epochs_test, test_sims['action_orien_cos_sim'], label='Test', alpha=0.7)
    plt.title('Orientation Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # 绘制综合性能指标 (损失越低越好，相似度越高越好)
    combined_train = []
    combined_test = []
    
    if (epochs_train and 'dist_loss' in train_losses and 'action_loss' in train_losses and 
        'action_waypts_cos_sim' in train_sims and 'action_orien_cos_sim' in train_sims):
        for i in range(min(len(train_losses['dist_loss']), len(train_losses['action_loss']), 
                          len(train_sims['action_waypts_cos_sim']), len(train_sims['action_orien_cos_sim']))):
            # 综合指标：相似度越高越好，损失越低越好
            score = (train_sims['action_waypts_cos_sim'][i] + train_sims['action_orien_cos_sim'][i]) / 2 - \
                   (train_losses['dist_loss'][i] + train_losses['action_loss'][i]) / 100
            combined_train.append(score)
    
    if (epochs_test and 'dist_loss' in test_losses and 'action_loss' in test_losses and 
        'action_waypts_cos_sim' in test_sims and 'action_orien_cos_sim' in test_sims):
        for i in range(min(len(test_losses['dist_loss']), len(test_losses['action_loss']), 
                          len(test_sims['action_waypts_cos_sim']), len(test_sims['action_orien_cos_sim']))):
            score = (test_sims['action_waypts_cos_sim'][i] + test_sims['action_orien_cos_sim'][i]) / 2 - \
                   (test_losses['dist_loss'][i] + test_losses['action_loss'][i]) / 100
            combined_test.append(score)
    
    if combined_train:
        plt.plot(range(len(combined_train)), combined_train, label='Train Combined', alpha=0.7)
    if combined_test:
        plt.plot(range(len(combined_test)), combined_test, label='Test Combined', alpha=0.7)
    plt.title('Combined Performance Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves have been saved to '{output_dir}' directory")

def main():
    # 检查当前目录下的日志文件
    print("正在检查当前目录下的日志文件...")
    log_files = [f for f in os.listdir('.') if f.endswith('.log') and 'train' in f.lower()]
    
    if log_files:
        print(f"找到以下日志文件: {log_files}")
        log_file_path = input(f"请输入日志文件名 (默认: {log_files[0]}): ").strip()
        if not log_file_path:
            log_file_path = log_files[0]
    else:
        log_file_path = input("请输入你的训练日志文件路径: ").strip()
    
    # 如果没有路径分隔符，假设在当前目录
    if '/' not in log_file_path and '\\' not in log_file_path:
        log_file_path = './' + log_file_path
    
    if not os.path.exists(log_file_path):
        print(f"错误: 找不到文件 {log_file_path}")
        print(f"当前目录文件: {os.listdir('.')}")
        return
    
    print("正在解析训练日志...")
    train_losses, train_sims, test_losses, test_sims = parse_complete_log(log_file_path)
    
    print("正在绘制训练曲线...")
    plot_training_curves(train_losses, train_sims, test_losses, test_sims)
    
    # 打印一些统计信息
    print("\n训练统计信息:")
    max_train_epochs = max([len(v) for v in train_losses.values() if v] + [len(v) for v in train_sims.values() if v] + [0])
    max_test_epochs = max([len(v) for v in test_losses.values() if v] + [len(v) for v in test_sims.values() if v] + [0])
    
    print(f"训练轮次: {max_train_epochs}")
    print(f"测试轮次: {max_test_epochs}")
    
    if test_losses and 'dist_loss' in test_losses and len(test_losses['dist_loss']) > 0:
        if test_losses['dist_loss']:
            best_epoch = np.argmin(test_losses['dist_loss'])
            print(f"最佳距离损失: {min(test_losses['dist_loss']):.4f} (第 {best_epoch} 轮)")
    
    if test_sims and 'action_waypts_cos_sim' in test_sims and len(test_sims['action_waypts_cos_sim']) > 0:
        if test_sims['action_waypts_cos_sim']:
            best_epoch = np.argmax(test_sims['action_waypts_cos_sim'])
            print(f"最佳路径点相似度: {max(test_sims['action_waypts_cos_sim']):.4f} (第 {best_epoch} 轮)")

if __name__ == "__main__":
    main()