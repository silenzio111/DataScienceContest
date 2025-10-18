"""
评估合成数据质量
对比真实数据和合成数据的分布差异
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估合成数据质量')

    parser.add_argument('--real-data', type=str,
                       default='../初赛选手数据/训练数据集.xlsx',
                       help='真实数据路径')

    parser.add_argument('--synthetic-data', type=str,
                       required=True,
                       help='合成数据路径')

    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='评估结果输出目录')

    return parser.parse_args()


def load_data(real_path, synthetic_path):
    """
    加载真实数据和合成数据

    Args:
        real_path: 真实数据路径
        synthetic_path: 合成数据路径

    Returns:
        real_data, synthetic_data
    """
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)

    # 加载真实数据
    if real_path.endswith('.xlsx'):
        real_data = pd.read_excel(real_path)
    else:
        real_data = pd.read_csv(real_path)
    print(f"真实数据: {real_data.shape}")

    # 加载合成数据
    if synthetic_path.endswith('.xlsx'):
        synthetic_data = pd.read_excel(synthetic_path)
    else:
        synthetic_data = pd.read_csv(synthetic_path)
    print(f"合成数据: {synthetic_data.shape}")

    return real_data, synthetic_data


def evaluate_distributions(real_data, synthetic_data):
    """
    评估数值特征的分布差异

    Args:
        real_data: 真实数据
        synthetic_data: 合成数据

    Returns:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("评估特征分布")
    print("="*60)

    results = []

    # 获取数值列
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in synthetic_data.columns:
            # 真实数据统计
            real_mean = real_data[col].mean()
            real_std = real_data[col].std()
            real_min = real_data[col].min()
            real_max = real_data[col].max()

            # 合成数据统计
            synth_mean = synthetic_data[col].mean()
            synth_std = synthetic_data[col].std()
            synth_min = synthetic_data[col].min()
            synth_max = synthetic_data[col].max()

            # KS检验 (Kolmogorov-Smirnov)
            ks_stat, ks_pvalue = stats.ks_2samp(
                real_data[col].dropna(),
                synthetic_data[col].dropna()
            )

            results.append({
                '特征': col,
                '真实均值': real_mean,
                '合成均值': synth_mean,
                '均值差异%': abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100,
                '真实标准差': real_std,
                '合成标准差': synth_std,
                'KS统计量': ks_stat,
                'KS p值': ks_pvalue,
                '分布相似': 'Yes' if ks_pvalue > 0.05 else 'No'
            })

    results_df = pd.DataFrame(results)

    print(f"\n总特征数: {len(results_df)}")
    print(f"分布相似特征数: {(results_df['分布相似']=='Yes').sum()}")
    print(f"分布相似比例: {(results_df['分布相似']=='Yes').mean():.2%}")

    return results_df


def evaluate_correlations(real_data, synthetic_data):
    """
    评估特征相关性

    Args:
        real_data: 真实数据
        synthetic_data: 合成数据

    Returns:
        corr_diff: 相关性差异矩阵
    """
    print("\n" + "="*60)
    print("评估特征相关性")
    print("="*60)

    # 获取数值列
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    common_cols = [col for col in numeric_cols if col in synthetic_data.columns]

    # 计算相关性矩阵
    real_corr = real_data[common_cols].corr()
    synth_corr = synthetic_data[common_cols].corr()

    # 计算差异
    corr_diff = np.abs(real_corr - synth_corr)

    print(f"平均相关性差异: {corr_diff.mean().mean():.4f}")
    print(f"最大相关性差异: {corr_diff.max().max():.4f}")

    return real_corr, synth_corr, corr_diff


def plot_feature_distributions(real_data, synthetic_data, output_dir, top_n=10):
    """
    绘制特征分布对比图

    Args:
        real_data: 真实数据
        synthetic_data: 合成数据
        output_dir: 输出目录
        top_n: 绘制前N个特征
    """
    print("\n" + "="*60)
    print(f"绘制前{top_n}个特征分布对比")
    print("="*60)

    # 获取数值列
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    common_cols = [col for col in numeric_cols if col in synthetic_data.columns]

    # 只选择前top_n个特征
    plot_cols = common_cols[:top_n]

    n_cols = 3
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]

        # 绘制分布
        ax.hist(real_data[col].dropna(), bins=30, alpha=0.5,
                label='Real', density=True, color='blue')
        ax.hist(synthetic_data[col].dropna(), bins=30, alpha=0.5,
                label='Synthetic', density=True, color='red')

        ax.set_title(f'{col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(plot_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 特征分布图已保存: {plot_path}")
    plt.close()


def plot_correlation_heatmaps(real_corr, synth_corr, corr_diff, output_dir):
    """
    绘制相关性热力图

    Args:
        real_corr: 真实数据相关性矩阵
        synth_corr: 合成数据相关性矩阵
        corr_diff: 相关性差异矩阵
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("绘制相关性热力图")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 真实数据相关性
    sns.heatmap(real_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Real Data Correlation')

    # 合成数据相关性
    sns.heatmap(synth_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Synthetic Data Correlation')

    # 相关性差异
    sns.heatmap(corr_diff, cmap='Reds', vmin=0, vmax=1,
                square=True, ax=axes[2], cbar_kws={'label': 'Absolute Difference'})
    axes[2].set_title('Correlation Difference')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'correlation_heatmaps.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 相关性热力图已保存: {plot_path}")
    plt.close()


def plot_pca_comparison(real_data, synthetic_data, output_dir):
    """
    绘制PCA降维对比图

    Args:
        real_data: 真实数据
        synthetic_data: 合成数据
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("绘制PCA降维对比")
    print("="*60)

    # 获取数值列
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    common_cols = [col for col in numeric_cols if col in synthetic_data.columns]

    # 准备数据
    real_features = real_data[common_cols].fillna(0)
    synth_features = synthetic_data[common_cols].fillna(0)

    # 标准化
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_features)
    synth_scaled = scaler.transform(synth_features)

    # PCA
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_scaled)
    synth_pca = pca.transform(synth_scaled)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=20)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthetic', s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA: Real vs Synthetic Data')
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = os.path.join(output_dir, 'pca_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ PCA对比图已保存: {plot_path}")
    plt.close()


def generate_evaluation_report(dist_results, real_corr, synth_corr, corr_diff, output_dir):
    """
    生成评估报告

    Args:
        dist_results: 分布评估结果
        real_corr: 真实数据相关性
        synth_corr: 合成数据相关性
        corr_diff: 相关性差异
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("生成评估报告")
    print("="*60)

    report_path = os.path.join(output_dir, 'evaluation_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 合成数据质量评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 总体评估
        f.write("## 总体评估\n\n")
        f.write(f"- 评估特征数: {len(dist_results)}\n")
        f.write(f"- 分布相似特征数: {(dist_results['分布相似']=='Yes').sum()}\n")
        f.write(f"- 分布相似比例: {(dist_results['分布相似']=='Yes').mean():.2%}\n")
        f.write(f"- 平均KS统计量: {dist_results['KS统计量'].mean():.4f}\n")
        f.write(f"- 平均相关性差异: {corr_diff.mean().mean():.4f}\n\n")

        # 分布评估详细结果
        f.write("---\n\n")
        f.write("## 特征分布评估\n\n")
        f.write(dist_results.to_markdown(index=False))
        f.write("\n\n")

        # 相关性评估
        f.write("---\n\n")
        f.write("## 相关性评估\n\n")
        f.write(f"- 平均相关性差异: {corr_diff.mean().mean():.4f}\n")
        f.write(f"- 最大相关性差异: {corr_diff.max().max():.4f}\n\n")

        # 结论
        f.write("---\n\n")
        f.write("## 结论\n\n")

        similarity_ratio = (dist_results['分布相似']=='Yes').mean()
        avg_corr_diff = corr_diff.mean().mean()

        if similarity_ratio > 0.8 and avg_corr_diff < 0.1:
            f.write("✅ **质量优秀**: 合成数据与真实数据高度相似，可以用于模型训练。\n\n")
        elif similarity_ratio > 0.6 and avg_corr_diff < 0.2:
            f.write("⚠️ **质量良好**: 合成数据与真实数据较为相似，可以谨慎使用。\n\n")
        else:
            f.write("❌ **质量较差**: 合成数据与真实数据差异较大，建议重新训练GAN模型。\n\n")

        f.write("---\n\n")

    print(f"✓ 评估报告已保存: {report_path}")

    # 保存详细结果
    dist_results.to_csv(os.path.join(output_dir, 'distribution_results.csv'), index=False)
    print(f"✓ 分布评估结果已保存: {os.path.join(output_dir, 'distribution_results.csv')}")


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("    合成数据质量评估")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    real_data, synthetic_data = load_data(args.real_data, args.synthetic_data)

    # 评估分布
    dist_results = evaluate_distributions(real_data, synthetic_data)

    # 评估相关性
    real_corr, synth_corr, corr_diff = evaluate_correlations(real_data, synthetic_data)

    # 生成可视化
    plot_feature_distributions(real_data, synthetic_data, args.output_dir)
    plot_correlation_heatmaps(real_corr, synth_corr, corr_diff, args.output_dir)
    plot_pca_comparison(real_data, synthetic_data, args.output_dir)

    # 生成报告
    generate_evaluation_report(dist_results, real_corr, synth_corr, corr_diff, args.output_dir)

    # 完成
    print("\n" + "="*60)
    print("✓ 评估完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"评估结果保存在: {args.output_dir}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断评估")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
