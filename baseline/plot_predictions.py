"""
预测值分布图绘制脚本
为所有submission文件生成预测值分布图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_submission_data(file_path):
    """加载submission文件数据"""
    try:
        df = pd.read_csv(file_path)
        if 'target' in df.columns:
            return df['target'].values
        else:
            print(f"警告: {file_path} 没有target列")
            return None
    except Exception as e:
        print(f"错误: 无法读取 {file_path}: {e}")
        return None

def plot_single_distribution(predictions, file_name, save_dir):
    """绘制单个文件的预测分布图"""
    if predictions is None:
        return

    plt.figure(figsize=(12, 8))

    # 创建子图
    gs = GridSpec(2, 2, figure=plt.gcf())

    # 1. 直方图
    ax1 = plt.subplot(gs[0, :])
    ax1.hist(predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'{file_name} - 预测概率分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测概率', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    mean_val = np.mean(predictions)
    median_val = np.median(predictions)
    std_val = np.std(predictions)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.4f}')
    ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'中位数: {median_val:.4f}')
    ax1.legend()

    # 2. 箱线图
    ax2 = plt.subplot(gs[1, 0])
    ax2.boxplot(predictions, vert=True, patch_artist=True)
    ax2.set_title('箱线图', fontsize=12)
    ax2.set_ylabel('预测概率', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 风险分布饼图
    ax3 = plt.subplot(gs[1, 1])
    risk_categories = [
        (predictions < 0.1).sum(),
        ((predictions >= 0.1) & (predictions < 0.3)).sum(),
        ((predictions >= 0.3) & (predictions < 0.5)).sum(),
        (predictions >= 0.5).sum()
    ]
    risk_labels = ['低风险(<0.1)', '中低风险(0.1-0.3)', '中高风险(0.3-0.5)', '高风险(≥0.5)']
    colors = ['lightgreen', 'yellow', 'orange', 'red']

    # 只显示非零的类别
    non_zero_indices = [i for i, count in enumerate(risk_categories) if count > 0]
    non_zero_categories = [risk_categories[i] for i in non_zero_indices]
    non_zero_labels = [risk_labels[i] for i in non_zero_indices]
    non_zero_colors = [colors[i] for i in non_zero_indices]

    if non_zero_categories:
        ax3.pie(non_zero_categories, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('风险分布', fontsize=12)

    plt.tight_layout()

    # 保存图片
    safe_filename = file_name.replace('.csv', '.png').replace('/', '_').replace('\\', '_')
    save_path = os.path.join(save_dir, safe_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': np.min(predictions),
        'max': np.max(predictions),
        'file_name': file_name
    }

def create_comparison_plot(all_stats, save_dir):
    """创建所有模型的对比图"""
    if not all_stats:
        return

    # 提取数据用于对比
    names = [stat['file_name'].replace('_submission.csv', '') for stat in all_stats]
    means = [stat['mean'] for stat in all_stats]
    medians = [stat['median'] for stat in all_stats]
    stds = [stat['std'] for stat in all_stats]

    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 均值对比
    bars1 = ax1.barh(range(len(names)), means, color='skyblue', alpha=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('平均预测概率', fontsize=12)
    ax1.set_title('各模型平均预测概率对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, (bar, value) in enumerate(zip(bars1, means)):
        ax1.text(value + max(means) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)

    # 2. 中位数对比
    bars2 = ax2.barh(range(len(names)), medians, color='lightgreen', alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel('中位数预测概率', fontsize=12)
    ax2.set_title('各模型中位数预测概率对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars2, medians)):
        ax2.text(value + max(medians) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)

    # 3. 标准差对比
    bars3 = ax3.barh(range(len(names)), stds, color='orange', alpha=0.7)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=10)
    ax3.set_xlabel('预测概率标准差', fontsize=12)
    ax3.set_title('各模型预测稳定性对比', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars3, stds)):
        ax3.text(value + max(stds) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)

    # 4. 均值vs中位数散点图
    ax4.scatter(means, medians, alpha=0.7, s=100, c='red')
    ax4.plot([0, max(means)], [0, max(means)], 'k--', alpha=0.5)  # 对角线

    # 添加标签
    for i, name in enumerate(names):
        ax4.annotate(name, (means[i], medians[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('平均预测概率', fontsize=12)
    ax4.set_ylabel('中位数预测概率', fontsize=12)
    ax4.set_title('均值 vs 中位数散点图', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_stats, save_dir):
    """创建汇总表格"""
    if not all_stats:
        return

    # 按均值排序
    sorted_stats = sorted(all_stats, key=lambda x: x['mean'], reverse=True)

    # 创建表格数据
    table_data = []
    for stat in sorted_stats:
        table_data.append([
            stat['file_name'].replace('_submission.csv', ''),
            f"{stat['mean']:.4f}",
            f"{stat['median']:.4f}",
            f"{stat['std']:.4f}",
            f"{stat['min']:.4f}",
            f"{stat['max']:.4f}"
        ])

    # 创建DataFrame
    df = pd.DataFrame(table_data,
                      columns=['模型名称', '均值', '中位数', '标准差', '最小值', '最大值'])

    # 保存为CSV
    df.to_csv(os.path.join(save_dir, 'predictions_summary.csv'), index=False, encoding='utf-8-sig')

    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(16, len(sorted_stats) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center', colColours=['#f0f0f0']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置标题
    plt.title('预测值分布汇总表', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(os.path.join(save_dir, 'predictions_summary_table.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("=== 开始绘制预测值分布图 ===")

    # 创建输出目录
    plots_dir = 'outputs/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # 获取所有submission文件
    submission_files = glob.glob('outputs/*submission.csv')

    if not submission_files:
        print("没有找到submission文件!")
        return

    print(f"找到 {len(submission_files)} 个submission文件")

    all_stats = []

    # 为每个文件绘制分布图
    for file_path in sorted(submission_files):
        file_name = os.path.basename(file_path)
        print(f"处理文件: {file_name}")

        predictions = load_submission_data(file_path)
        if predictions is not None:
            stats = plot_single_distribution(predictions, file_name, plots_dir)
            if stats:
                all_stats.append(stats)
                print(f"  ✓ 完成: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
            else:
                print(f"  ✗ 失败: 无法获取统计信息")
        else:
            print(f"  ✗ 失败: 无法加载数据")

    if all_stats:
        print(f"\n=== 创建对比分析图 ===")

        # 创建对比图
        create_comparison_plot(all_stats, plots_dir)
        print("✓ 模型对比图已保存")

        # 创建汇总表
        create_summary_table(all_stats, plots_dir)
        print("✓ 汇总表格已保存")

        print(f"\n=== 统计摘要 ===")
        sorted_by_mean = sorted(all_stats, key=lambda x: x['mean'], reverse=True)
        print(f"平均概率最高的模型: {sorted_by_mean[0]['file_name']} ({sorted_by_mean[0]['mean']:.4f})")
        print(f"平均概率最低的模型: {sorted_by_mean[-1]['file_name']} ({sorted_by_mean[-1]['mean']:.4f})")

        sorted_by_std = sorted(all_stats, key=lambda x: x['std'])
        print(f"预测最稳定的模型: {sorted_by_std[0]['file_name']} (标准差={sorted_by_std[0]['std']:.4f})")
        print(f"预测最不稳定的模型: {sorted_by_std[-1]['file_name']} (标准差={sorted_by_std[-1]['std']:.4f})")

    print(f"\n=== 完成 ===")
    print(f"所有图片已保存到: {plots_dir}/")
    print(f"生成的文件:")
    print(f"- 各模型分布图: {len(all_stats)} 个")
    print(f"- 模型对比图: models_comparison.png")
    print(f"- 汇总表格: predictions_summary.csv")
    print(f"- 可视化表格: predictions_summary_table.png")

if __name__ == "__main__":
    main()