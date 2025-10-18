"""
GAN模型训练脚本
使用CTGAN训练表格数据生成模型
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from ctgan import CTGAN
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练GAN模型生成合成数据')

    parser.add_argument('--train-path', type=str,
                       default='../初赛选手数据/训练数据集.xlsx',
                       help='训练数据路径')

    parser.add_argument('--target-class', type=str,
                       default='both',
                       choices=['both', 'minority', 'majority'],
                       help='训练目标：both(全部), minority(只违约), majority(只正常)')

    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数 (默认: 300)')

    parser.add_argument('--batch-size', type=int, default=500,
                       help='批次大小 (默认: 500)')

    parser.add_argument('--output-dir', type=str, default='models',
                       help='模型保存目录')

    parser.add_argument('--use-sdv', action='store_true',
                       help='使用SDV的CTGAN（推荐）')

    return parser.parse_args()


def load_and_prepare_data(train_path, target_class='both'):
    """
    加载和准备数据

    Args:
        train_path: 训练数据路径
        target_class: 训练目标类别

    Returns:
        train_data: 训练数据
        metadata: 数据元信息
    """
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)

    # 读取数据
    df = pd.read_excel(train_path)

    print(f"原始数据形状: {df.shape}")
    print(f"违约率: {df['target'].mean():.2%}")
    print(f"违约样本数: {df['target'].sum()}")
    print(f"正常样本数: {(df['target']==0).sum()}")

    # 根据目标类别过滤数据
    if target_class == 'minority':
        train_data = df[df['target'] == 1].copy()
        print(f"\n只训练违约样本: {len(train_data)} 条")
    elif target_class == 'majority':
        train_data = df[df['target'] == 0].copy()
        print(f"\n只训练正常样本: {len(train_data)} 条")
    else:
        train_data = df.copy()
        print(f"\n训练全部样本: {len(train_data)} 条")

    return train_data


def create_metadata(train_data):
    """
    创建数据元信息

    Args:
        train_data: 训练数据

    Returns:
        metadata: SDV元数据对象
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)

    # 确保target被识别为分类变量
    metadata.update_column(
        column_name='target',
        sdtype='categorical'
    )

    return metadata


def train_ctgan_sdv(train_data, metadata, epochs=300, batch_size=500):
    """
    使用SDV的CTGAN训练（推荐）

    Args:
        train_data: 训练数据
        metadata: 数据元信息
        epochs: 训练轮数
        batch_size: 批次大小

    Returns:
        synthesizer: 训练好的合成器
    """
    print("\n" + "="*60)
    print("使用SDV CTGAN训练模型")
    print("="*60)
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")

    # 创建合成器
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    # 训练
    print("\n开始训练...")
    synthesizer.fit(train_data)
    print("✓ 训练完成!")

    return synthesizer


def train_ctgan_original(train_data, epochs=300, batch_size=500):
    """
    使用原始CTGAN训练

    Args:
        train_data: 训练数据
        epochs: 训练轮数
        batch_size: 批次大小

    Returns:
        model: 训练好的CTGAN模型
    """
    print("\n" + "="*60)
    print("使用原始CTGAN训练模型")
    print("="*60)
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")

    # 识别离散列
    discrete_columns = ['target']

    # 创建模型
    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    # 训练
    print("\n开始训练...")
    model.fit(train_data, discrete_columns)
    print("✓ 训练完成!")

    return model


def save_model(model, output_dir, target_class, use_sdv):
    """
    保存训练好的模型

    Args:
        model: 训练好的模型
        output_dir: 输出目录
        target_class: 目标类别
        use_sdv: 是否使用SDV
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "sdv_ctgan" if use_sdv else "ctgan"
    model_name = f"{model_type}_{target_class}_{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_name)

    print(f"\n保存模型到: {model_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # 保存最新模型链接
    latest_path = os.path.join(output_dir, f"{model_type}_{target_class}_latest.pkl")
    with open(latest_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ 模型已保存")
    print(f"  完整模型: {model_name}")
    print(f"  最新链接: {model_type}_{target_class}_latest.pkl")


def generate_test_samples(model, num_samples, use_sdv):
    """
    生成测试样本验证模型

    Args:
        model: 训练好的模型
        num_samples: 生成样本数
        use_sdv: 是否使用SDV

    Returns:
        synthetic_data: 合成数据
    """
    print(f"\n生成 {num_samples} 条测试样本...")

    if use_sdv:
        synthetic_data = model.sample(num_rows=num_samples)
    else:
        synthetic_data = model.sample(num_samples)

    print(f"✓ 生成完成")
    print(f"  形状: {synthetic_data.shape}")
    print(f"  违约率: {synthetic_data['target'].mean():.2%}")

    return synthetic_data


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("    GAN模型训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查训练数据
    if not os.path.exists(args.train_path):
        print(f"\n❌ 错误: 找不到训练数据: {args.train_path}")
        return 1

    # 加载数据
    train_data = load_and_prepare_data(args.train_path, args.target_class)

    # 训练模型
    if args.use_sdv:
        # 使用SDV CTGAN（推荐）
        metadata = create_metadata(train_data)
        model = train_ctgan_sdv(
            train_data,
            metadata,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # 使用原始CTGAN
        model = train_ctgan_original(
            train_data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # 保存模型
    save_model(model, args.output_dir, args.target_class, args.use_sdv)

    # 生成测试样本
    test_samples = generate_test_samples(model, 100, args.use_sdv)

    # 保存测试样本
    test_output_dir = os.path.join(args.output_dir, 'test_samples')
    os.makedirs(test_output_dir, exist_ok=True)
    test_path = os.path.join(test_output_dir, f'test_samples_{args.target_class}.csv')
    test_samples.to_csv(test_path, index=False)
    print(f"  测试样本已保存: {test_path}")

    # 完成
    print("\n" + "="*60)
    print("✓ 训练完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型保存在: {args.output_dir}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
