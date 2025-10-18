"""
生成合成数据脚本
使用训练好的GAN模型生成合成样本
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用GAN模型生成合成数据')

    parser.add_argument('--model-path', type=str,
                       help='模型文件路径 (如不指定则使用latest)')

    parser.add_argument('--model-dir', type=str, default='models',
                       help='模型目录')

    parser.add_argument('--target-class', type=str, default='both',
                       choices=['both', 'minority', 'majority'],
                       help='模型类型')

    parser.add_argument('--num-samples', type=int, default=500,
                       help='生成样本数 (默认: 500)')

    parser.add_argument('--output-dir', type=str, default='synthetic_data',
                       help='输出目录')

    parser.add_argument('--output-name', type=str, default=None,
                       help='输出文件名 (不包含扩展名)')

    parser.add_argument('--use-sdv', action='store_true',
                       help='使用SDV的CTGAN模型')

    return parser.parse_args()


def load_model(model_path=None, model_dir='models', target_class='both', use_sdv=False):
    """
    加载训练好的模型

    Args:
        model_path: 模型文件路径
        model_dir: 模型目录
        target_class: 目标类别
        use_sdv: 是否使用SDV模型

    Returns:
        model: 加载的模型
    """
    print("\n" + "="*60)
    print("加载模型")
    print("="*60)

    if model_path is None:
        # 使用latest模型
        model_type = "sdv_ctgan" if use_sdv else "ctgan"
        model_path = os.path.join(model_dir, f"{model_type}_{target_class}_latest.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    print(f"模型路径: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("✓ 模型加载成功")

    return model


def generate_synthetic_data(model, num_samples, use_sdv):
    """
    生成合成数据

    Args:
        model: 训练好的模型
        num_samples: 生成样本数
        use_sdv: 是否使用SDV模型

    Returns:
        synthetic_data: 合成数据
    """
    print("\n" + "="*60)
    print("生成合成数据")
    print("="*60)
    print(f"生成样本数: {num_samples}")

    if use_sdv:
        synthetic_data = model.sample(num_rows=num_samples)
    else:
        synthetic_data = model.sample(num_samples)

    print(f"✓ 生成完成")
    print(f"  形状: {synthetic_data.shape}")

    # 统计信息
    if 'target' in synthetic_data.columns:
        print(f"  违约样本数: {synthetic_data['target'].sum()}")
        print(f"  正常样本数: {(synthetic_data['target']==0).sum()}")
        print(f"  违约率: {synthetic_data['target'].mean():.2%}")

    return synthetic_data


def save_synthetic_data(synthetic_data, output_dir, output_name=None):
    """
    保存合成数据

    Args:
        synthetic_data: 合成数据
        output_dir: 输出目录
        output_name: 输出文件名
    """
    os.makedirs(output_dir, exist_ok=True)

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"synthetic_data_{timestamp}"

    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    excel_path = os.path.join(output_dir, f"{output_name}.xlsx")

    print("\n保存合成数据...")

    # 保存CSV
    synthetic_data.to_csv(csv_path, index=False)
    print(f"✓ CSV已保存: {csv_path}")

    # 保存Excel
    synthetic_data.to_excel(excel_path, index=False)
    print(f"✓ Excel已保存: {excel_path}")


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("    生成合成数据")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型
    model = load_model(
        model_path=args.model_path,
        model_dir=args.model_dir,
        target_class=args.target_class,
        use_sdv=args.use_sdv
    )

    # 生成合成数据
    synthetic_data = generate_synthetic_data(
        model,
        args.num_samples,
        args.use_sdv
    )

    # 保存合成数据
    save_synthetic_data(
        synthetic_data,
        args.output_dir,
        args.output_name
    )

    # 完成
    print("\n" + "="*60)
    print("✓ 生成完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output_dir}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断生成")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 生成出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
