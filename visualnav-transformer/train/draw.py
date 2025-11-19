import re
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def parse_log_file(log_file_path):
    """
    解析单个训练日志文件
    """
    if not os.path.exists(log_file_path):
        print(f"错误: 找不到文件 {log_file_path}")
        return None
    
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
    
    return {
        'train_losses': train_losses,
        'train_sims': train_sims,
        'test_losses': test_losses,
        'test_sims': test_sims
    }

def plot_comparison_curves(model1_data, model2_data, model1_name="Model 1", model2_name="Model 2", output_dir='comparison_plots'):
    """
    绘制两个模型的对比曲线
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取最大的epoch数
    max_epochs = 0
    
    # 为所有指标设置相同的x轴范围
    all_epochs = set()
    for data in [model1_data, model2_data]:
        for key in data['train_losses'].keys():
            if len(data['train_losses'][key]) > 0:
                all_epochs.update(range(len(data['train_losses'][key])))
        for key in data['test_losses'].keys():
            if len(data['test_losses'][key]) > 0:
                all_epochs.update(range(len(data['test_losses'][key])))
        for key in data['train_sims'].keys():
            if len(data['train_sims'][key]) > 0:
                all_epochs.update(range(len(data['train_sims'][key])))
        for key in data['test_sims'].keys():
            if len(data['test_sims'][key]) > 0:
                all_epochs.update(range(len(data['test_sims'][key])))
    
    x_range = list(all_epochs) if all_epochs else []
    
    # 1. 损失函数对比图
    plt.figure(figsize=(15, 10))
    
    # Distance Loss
    plt.subplot(2, 3, 1)
    if len(model1_data['train_losses']['dist_loss']) > 0:
        plt.plot(range(len(model1_data['train_losses']['dist_loss'])), 
                model1_data['train_losses']['dist_loss'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_losses']['dist_loss']) > 0:
        plt.plot(range(len(model1_data['test_losses']['dist_loss'])), 
                model1_data['test_losses']['dist_loss'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_losses']['dist_loss']) > 0:
        plt.plot(range(len(model2_data['train_losses']['dist_loss'])), 
                model2_data['train_losses']['dist_loss'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_losses']['dist_loss']) > 0:
        plt.plot(range(len(model2_data['test_losses']['dist_loss'])), 
                model2_data['test_losses']['dist_loss'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Distance Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Action Loss
    plt.subplot(2, 3, 2)
    if len(model1_data['train_losses']['action_loss']) > 0:
        plt.plot(range(len(model1_data['train_losses']['action_loss'])), 
                model1_data['train_losses']['action_loss'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_losses']['action_loss']) > 0:
        plt.plot(range(len(model1_data['test_losses']['action_loss'])), 
                model1_data['test_losses']['action_loss'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_losses']['action_loss']) > 0:
        plt.plot(range(len(model2_data['train_losses']['action_loss'])), 
                model2_data['train_losses']['action_loss'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_losses']['action_loss']) > 0:
        plt.plot(range(len(model2_data['test_losses']['action_loss'])), 
                model2_data['test_losses']['action_loss'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Action Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Total Loss
    plt.subplot(2, 3, 3)
    if len(model1_data['train_losses']['total_loss']) > 0:
        plt.plot(range(len(model1_data['train_losses']['total_loss'])), 
                model1_data['train_losses']['total_loss'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_losses']['total_loss']) > 0:
        plt.plot(range(len(model1_data['test_losses']['total_loss'])), 
                model1_data['test_losses']['total_loss'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_losses']['total_loss']) > 0:
        plt.plot(range(len(model2_data['train_losses']['total_loss'])), 
                model2_data['train_losses']['total_loss'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_losses']['total_loss']) > 0:
        plt.plot(range(len(model2_data['test_losses']['total_loss'])), 
                model2_data['test_losses']['total_loss'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Total Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Waypoints Similarity
    plt.subplot(2, 3, 4)
    if len(model1_data['train_sims']['action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model1_data['train_sims']['action_waypts_cos_sim'])), 
                model1_data['train_sims']['action_waypts_cos_sim'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_sims']['action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model1_data['test_sims']['action_waypts_cos_sim'])), 
                model1_data['test_sims']['action_waypts_cos_sim'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_sims']['action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model2_data['train_sims']['action_waypts_cos_sim'])), 
                model2_data['train_sims']['action_waypts_cos_sim'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_sims']['action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model2_data['test_sims']['action_waypts_cos_sim'])), 
                model2_data['test_sims']['action_waypts_cos_sim'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Waypoints Similarity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Orientation Similarity
    plt.subplot(2, 3, 5)
    if len(model1_data['train_sims']['action_orien_cos_sim']) > 0:
        plt.plot(range(len(model1_data['train_sims']['action_orien_cos_sim'])), 
                model1_data['train_sims']['action_orien_cos_sim'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_sims']['action_orien_cos_sim']) > 0:
        plt.plot(range(len(model1_data['test_sims']['action_orien_cos_sim'])), 
                model1_data['test_sims']['action_orien_cos_sim'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_sims']['action_orien_cos_sim']) > 0:
        plt.plot(range(len(model2_data['train_sims']['action_orien_cos_sim'])), 
                model2_data['train_sims']['action_orien_cos_sim'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_sims']['action_orien_cos_sim']) > 0:
        plt.plot(range(len(model2_data['test_sims']['action_orien_cos_sim'])), 
                model2_data['test_sims']['action_orien_cos_sim'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Orientation Similarity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Multi Action Waypoints Similarity
    plt.subplot(2, 3, 6)
    if len(model1_data['train_sims']['multi_action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model1_data['train_sims']['multi_action_waypts_cos_sim'])), 
                model1_data['train_sims']['multi_action_waypts_cos_sim'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_sims']['multi_action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model1_data['test_sims']['multi_action_waypts_cos_sim'])), 
                model1_data['test_sims']['multi_action_waypts_cos_sim'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_sims']['multi_action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model2_data['train_sims']['multi_action_waypts_cos_sim'])), 
                model2_data['train_sims']['multi_action_waypts_cos_sim'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_sims']['multi_action_waypts_cos_sim']) > 0:
        plt.plot(range(len(model2_data['test_sims']['multi_action_waypts_cos_sim'])), 
                model2_data['test_sims']['multi_action_waypts_cos_sim'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Multi Action Waypoints Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_similarity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建综合性能对比图
    plt.figure(figsize=(15, 5))
    
    # 综合性能指标对比
    plt.subplot(1, 2, 1)
    # 计算第一个模型的综合性能
    combined1_train = []
    combined1_test = []
    if (len(model1_data['train_losses']['dist_loss']) > 0 and len(model1_data['train_losses']['action_loss']) > 0 and 
        len(model1_data['train_sims']['action_waypts_cos_sim']) > 0 and len(model1_data['train_sims']['action_orien_cos_sim']) > 0):
        for i in range(min(len(model1_data['train_losses']['dist_loss']), len(model1_data['train_losses']['action_loss']), 
                         len(model1_data['train_sims']['action_waypts_cos_sim']), len(model1_data['train_sims']['action_orien_cos_sim']))):
            score = (model1_data['train_sims']['action_waypts_cos_sim'][i] + model1_data['train_sims']['action_orien_cos_sim'][i]) / 2 - \
                   (model1_data['train_losses']['dist_loss'][i] + model1_data['train_losses']['action_loss'][i]) / 100
            combined1_train.append(score)
    
    if (len(model1_data['test_losses']['dist_loss']) > 0 and len(model1_data['test_losses']['action_loss']) > 0 and 
        len(model1_data['test_sims']['action_waypts_cos_sim']) > 0 and len(model1_data['test_sims']['action_orien_cos_sim']) > 0):
        for i in range(min(len(model1_data['test_losses']['dist_loss']), len(model1_data['test_losses']['action_loss']), 
                         len(model1_data['test_sims']['action_waypts_cos_sim']), len(model1_data['test_sims']['action_orien_cos_sim']))):
            score = (model1_data['test_sims']['action_waypts_cos_sim'][i] + model1_data['test_sims']['action_orien_cos_sim'][i]) / 2 - \
                   (model1_data['test_losses']['dist_loss'][i] + model1_data['test_losses']['action_loss'][i]) / 100
            combined1_test.append(score)
    
    # 计算第二个模型的综合性能
    combined2_train = []
    combined2_test = []
    if (len(model2_data['train_losses']['dist_loss']) > 0 and len(model2_data['train_losses']['action_loss']) > 0 and 
        len(model2_data['train_sims']['action_waypts_cos_sim']) > 0 and len(model2_data['train_sims']['action_orien_cos_sim']) > 0):
        for i in range(min(len(model2_data['train_losses']['dist_loss']), len(model2_data['train_losses']['action_loss']), 
                         len(model2_data['train_sims']['action_waypts_cos_sim']), len(model2_data['train_sims']['action_orien_cos_sim']))):
            score = (model2_data['train_sims']['action_waypts_cos_sim'][i] + model2_data['train_sims']['action_orien_cos_sim'][i]) / 2 - \
                   (model2_data['train_losses']['dist_loss'][i] + model2_data['train_losses']['action_loss'][i]) / 100
            combined2_train.append(score)
    
    if (len(model2_data['test_losses']['dist_loss']) > 0 and len(model2_data['test_losses']['action_loss']) > 0 and 
        len(model2_data['test_sims']['action_waypts_cos_sim']) > 0 and len(model2_data['test_sims']['action_orien_cos_sim']) > 0):
        for i in range(min(len(model2_data['test_losses']['dist_loss']), len(model2_data['test_losses']['action_loss']), 
                         len(model2_data['test_sims']['action_waypts_cos_sim']), len(model2_data['test_sims']['action_orien_cos_sim']))):
            score = (model2_data['test_sims']['action_waypts_cos_sim'][i] + model2_data['test_sims']['action_orien_cos_sim'][i]) / 2 - \
                   (model2_data['test_losses']['dist_loss'][i] + model2_data['test_losses']['action_loss'][i]) / 100
            combined2_test.append(score)
    
    # 绘制综合性能
    if combined1_train:
        plt.plot(range(len(combined1_train)), combined1_train, label=f'{model1_name} Train Combined', alpha=0.7, linewidth=2)
    if combined1_test:
        plt.plot(range(len(combined1_test)), combined1_test, label=f'{model1_name} Test Combined', alpha=0.7, linewidth=2)
    if combined2_train:
        plt.plot(range(len(combined2_train)), combined2_train, label=f'{model2_name} Train Combined', alpha=0.7, linestyle='--', linewidth=2)
    if combined2_test:
        plt.plot(range(len(combined2_test)), combined2_test, label=f'{model2_name} Test Combined', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Combined Performance Metric Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 多动作方向相似度对比
    plt.subplot(1, 2, 2)
    if len(model1_data['train_sims']['multi_action_orien_cos_sim']) > 0:
        plt.plot(range(len(model1_data['train_sims']['multi_action_orien_cos_sim'])), 
                model1_data['train_sims']['multi_action_orien_cos_sim'], 
                label=f'{model1_name} Train', alpha=0.7, linewidth=2)
    if len(model1_data['test_sims']['multi_action_orien_cos_sim']) > 0:
        plt.plot(range(len(model1_data['test_sims']['multi_action_orien_cos_sim'])), 
                model1_data['test_sims']['multi_action_orien_cos_sim'], 
                label=f'{model1_name} Test', alpha=0.7, linewidth=2)
    if len(model2_data['train_sims']['multi_action_orien_cos_sim']) > 0:
        plt.plot(range(len(model2_data['train_sims']['multi_action_orien_cos_sim'])), 
                model2_data['train_sims']['multi_action_orien_cos_sim'], 
                label=f'{model2_name} Train', alpha=0.7, linestyle='--', linewidth=2)
    if len(model2_data['test_sims']['multi_action_orien_cos_sim']) > 0:
        plt.plot(range(len(model2_data['test_sims']['multi_action_orien_cos_sim'])), 
                model2_data['test_sims']['multi_action_orien_cos_sim'], 
                label=f'{model2_name} Test', alpha=0.7, linestyle='--', linewidth=2)
    plt.title('Multi Action Orientation Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots have been saved to '{output_dir}' directory")

def main():
    # 设置模型名称
    model1_name = "Mamba-ViNT"
    model2_name = "ViNT"
    
    # 获取当前目录下的日志文件
    print("正在检查当前目录下的日志文件...")
    log_files = [f for f in os.listdir('.') if f.endswith('.log') and 'train' in f.lower()]
    
    if not log_files:
        print("未找到任何日志文件")
        return
    
    # 自动识别两个模型的日志文件
    model1_log = None
    model2_log = None
    
    for log_file in log_files:
        if 'vint' in log_file.lower() and 'training' in log_file.lower():
            model2_log = log_file
        elif 'training' in log_file.lower() and 'vint' not in log_file.lower():
            model1_log = log_file
    
    # 如果没有自动识别成功，手动输入
    if not model1_log or not model2_log:
        print("未能自动识别两个模型的日志文件，请手动输入：")
        model1_log = input(f"请输入第一个模型的日志文件名 ({model1_name}): ").strip()
        model2_log = input(f"请输入第二个模型的日志文件名 ({model2_name}): ").strip()
    
    # 确保文件存在
    if not os.path.exists(model1_log):
        print(f"错误: 找不到文件 {model1_log}")
        return
    if not os.path.exists(model2_log):
        print(f"错误: 找不到文件 {model2_log}")
        return
    
    print(f"正在解析 {model1_name} 的日志文件: {model1_log}")
    model1_data = parse_log_file(model1_log)
    if model1_data is None:
        return
    
    print(f"正在解析 {model2_name} 的日志文件: {model2_log}")
    model2_data = parse_log_file(model2_log)
    if model2_data is None:
        return
    
    print("正在绘制对比曲线...")
    plot_comparison_curves(model1_data, model2_data, model1_name, model2_name)
    
    # 打印一些统计信息
    print("\n训练统计信息:")
    print(f"{model1_name}: 训练轮次 {max([len(v) for v in model1_data['train_losses'].values() if v] + [0])}")
    print(f"{model2_name}: 训练轮次 {max([len(v) for v in model2_data['train_losses'].values() if v] + [0])}")

if __name__ == "__main__":
    main()