"""
G-XGBoost训练脚本
使用GAN增强的数据训练XGBoost和Stacking模型
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# 使用GANmodel本地模块
from gan_data_utils import load_data, handle_missing_values, create_features, balance_samples
from gan_models import get_models, evaluate_model, make_predictions, train_stacking_v2


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='G-XGBoost: GAN增强的模型训练')

    parser.add_argument('--train-path', type=str,
                       default='../初赛选手数据/训练数据集.xlsx',
                       help='原始训练数据路径')

    parser.add_argument('--test-path', type=str,
                       default='../初赛选手数据/测试集.xlsx',
                       help='测试数据路径')

    parser.add_argument('--sample-path', type=str,
                       default='../初赛选手数据/提交样例.csv',
                       help='提交样例路径')

    parser.add_argument('--gan-model-path', type=str,
                       help='GAN模型路径')

    parser.add_argument('--synthetic-data-path', type=str,
                       help='预生成的合成数据路径（如提供则不使用GAN模型）')

    parser.add_argument('--num-synthetic', type=int, default=500,
                       help='生成合成样本数 (默认: 500)')

    parser.add_argument('--augment-strategy', type=str, default='minority',
                       choices=['minority', 'both', 'balanced'],
                       help='增强策略: minority(只增强违约), both(增强全部), balanced(平衡)')

    parser.add_argument('--model-type', type=str, default='stacking',
                       choices=['xgboost', 'stacking', 'ensemble'],
                       help='模型类型')

    parser.add_argument('--use-smote', action='store_true',
                       help='是否在增强后仍使用SMOTE')

    parser.add_argument('--output-name', type=str, default='g_xgboost',
                       help='输出文件名前缀')

    parser.add_argument('--use-sdv', action='store_true',
                       help='使用SDV的CTGAN模型')

    return parser.parse_args()


def load_or_generate_synthetic_data(gan_model_path, synthetic_data_path,
                                    num_synthetic, use_sdv):
    """
    加载或生成合成数据

    Args:
        gan_model_path: GAN模型路径
        synthetic_data_path: 预生成合成数据路径
        num_synthetic: 生成样本数
        use_sdv: 是否使用SDV模型

    Returns:
        synthetic_data: 合成数据
    """
    print("\n" + "="*60)
    print("获取合成数据")
    print("="*60)

    if synthetic_data_path and os.path.exists(synthetic_data_path):
        # 使用预生成的合成数据
        print(f"从文件加载合成数据: {synthetic_data_path}")

        if synthetic_data_path.endswith('.xlsx'):
            synthetic_data = pd.read_excel(synthetic_data_path)
        else:
            synthetic_data = pd.read_csv(synthetic_data_path)

        print(f"✓ 合成数据加载完成: {synthetic_data.shape}")

    elif gan_model_path and os.path.exists(gan_model_path):
        # 使用GAN模型生成
        print(f"从GAN模型生成合成数据: {gan_model_path}")
        print(f"生成样本数: {num_synthetic}")

        with open(gan_model_path, 'rb') as f:
            model = pickle.load(f)

        if use_sdv:
            synthetic_data = model.sample(num_rows=num_synthetic)
        else:
            synthetic_data = model.sample(num_synthetic)

        print(f"✓ 合成数据生成完成: {synthetic_data.shape}")

    else:
        raise ValueError("必须提供 --gan-model-path 或 --synthetic-data-path")

    # 显示统计信息
    if 'target' in synthetic_data.columns:
        print(f"  违约样本: {synthetic_data['target'].sum()}")
        print(f"  正常样本: {(synthetic_data['target']==0).sum()}")
        print(f"  违约率: {synthetic_data['target'].mean():.2%}")

    return synthetic_data


def augment_training_data(X_train, y_train, synthetic_data, strategy='minority'):
    """
    使用合成数据增强训练集

    Args:
        X_train: 原始训练特征
        y_train: 原始训练标签
        synthetic_data: 合成数据
        strategy: 增强策略

    Returns:
        X_augmented, y_augmented: 增强后的训练数据
    """
    print("\n" + "="*60)
    print("数据增强")
    print("="*60)
    print(f"增强策略: {strategy}")

    # 合并原始训练数据
    train_data = X_train.copy()
    train_data['target'] = y_train

    print(f"原始训练数据: {len(train_data)} 条")
    print(f"  违约: {(train_data['target']==1).sum()} 条")
    print(f"  正常: {(train_data['target']==0).sum()} 条")

    # 根据策略选择合成数据
    if strategy == 'minority':
        # 只使用违约样本
        synthetic_subset = synthetic_data[synthetic_data['target'] == 1].copy()
        print(f"\n使用合成违约样本: {len(synthetic_subset)} 条")

    elif strategy == 'both':
        # 使用全部合成样本
        synthetic_subset = synthetic_data.copy()
        print(f"\n使用全部合成样本: {len(synthetic_subset)} 条")

    elif strategy == 'balanced':
        # 平衡采样
        n_minority = (train_data['target']==1).sum()
        n_majority = (train_data['target']==0).sum()

        # 计算需要的样本数
        target_minority = max(n_minority, n_majority)
        n_synth_minority = target_minority - n_minority

        # 采样合成数据
        synth_minority = synthetic_data[synthetic_data['target'] == 1].sample(
            n=min(n_synth_minority, (synthetic_data['target']==1).sum()),
            replace=True
        )

        synthetic_subset = synth_minority
        print(f"\n使用平衡策略")
        print(f"  原始违约: {n_minority}")
        print(f"  原始正常: {n_majority}")
        print(f"  合成违约: {len(synthetic_subset)}")

    else:
        raise ValueError(f"未知的增强策略: {strategy}")

    # 合并数据
    augmented_data = pd.concat([train_data, synthetic_subset], ignore_index=True)

    print(f"\n增强后训练数据: {len(augmented_data)} 条")
    print(f"  违约: {(augmented_data['target']==1).sum()} 条")
    print(f"  正常: {(augmented_data['target']==0).sum()} 条")
    print(f"  违约率: {augmented_data['target'].mean():.2%}")

    # 分离特征和标签
    y_augmented = augmented_data['target']
    X_augmented = augmented_data.drop('target', axis=1)

    return X_augmented, y_augmented


def train_model(X_train, y_train, X_test, model_type='stacking'):
    """
    训练模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        model_type: 模型类型

    Returns:
        predictions: 预测结果
        model_info: 模型信息
    """
    print("\n" + "="*60)
    print(f"训练{model_type}模型")
    print("="*60)

    if model_type == 'xgboost':
        # 训练单个XGBoost
        models = get_models()
        xgb_model = models['xgboost']
        trained_model, metrics = evaluate_model(xgb_model, X_train, y_train, 'xgboost')

        predictions = trained_model.predict_proba(X_test)[:, 1]
        model_info = {
            'type': 'xgboost',
            'metrics': metrics
        }

    elif model_type == 'stacking':
        # 训练Stacking模型
        stacking_model, metrics = train_stacking_v2(
            X_train, y_train,
            strategy='simple_average',
            top_n=3,
            n_folds=5
        )

        predictions = stacking_model.predict_proba(X_test)[:, 1]
        model_info = {
            'type': 'stacking',
            'metrics': metrics
        }

    elif model_type == 'ensemble':
        # 训练Ensemble
        models = get_models()

        # Top 3模型
        top_models = ['xgboost', 'random_forest', 'gradient_boosting']
        all_predictions = []

        for model_name in top_models:
            print(f"\n训练 {model_name}...")
            model = models[model_name]
            trained_model, metrics = evaluate_model(model, X_train, y_train, model_name)

            preds = trained_model.predict_proba(X_test)[:, 1]
            all_predictions.append(preds)

        # 简单平均
        predictions = np.mean(all_predictions, axis=0)
        model_info = {
            'type': 'ensemble',
            'models': top_models
        }

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    return predictions, model_info


def save_predictions(predictions, sample_path, output_name):
    """
    保存预测结果

    Args:
        predictions: 预测结果
        sample_path: 提交样例路径
        output_name: 输出文件名
    """
    print("\n" + "="*60)
    print("保存预测结果")
    print("="*60)

    # 读取提交样例
    if os.path.exists(sample_path):
        submission = pd.read_csv(sample_path)
    else:
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'target': [0] * len(predictions)
        })

    submission['target'] = predictions

    # 保存到GANmodel自己的输出目录
    os.makedirs('outputs', exist_ok=True)
    output_path = f'outputs/{output_name}_submission.csv'
    submission.to_csv(output_path, index=False)

    print(f"✓ 预测文件已保存: {output_path}")
    print(f"  预测均值: {predictions.mean():.4f}")
    print(f"  预测标准差: {predictions.std():.4f}")
    print(f"  预测最小值: {predictions.min():.4f}")
    print(f"  预测最大值: {predictions.max():.4f}")

    return output_path


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("    G-XGBoost: GAN增强的模型训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查文件
    for path, name in [(args.train_path, "训练集"), (args.test_path, "测试集")]:
        if not os.path.exists(path):
            print(f"\n❌ 错误: 找不到{name}文件: {path}")
            return 1

    # 步骤1: 加载和预处理原始数据
    print("\n" + "="*60)
    print("步骤 1: 加载原始数据")
    print("="*60)

    # 加载原始数据
    train_df = load_data(args.train_path)
    test_df = load_data(args.test_path)

    # 处理缺失值
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # 特征工程
    X_train_original, X_test, y_train_original = create_features(train_df, test_df)

    print(f"原始训练数据: {X_train_original.shape}")
    print(f"测试数据: {X_test.shape}")

    # 步骤2: 获取合成数据
    synthetic_data = load_or_generate_synthetic_data(
        args.gan_model_path,
        args.synthetic_data_path,
        args.num_synthetic,
        args.use_sdv
    )

    # 步骤3: 数据增强
    X_train_augmented, y_train_augmented = augment_training_data(
        X_train_original,
        y_train_original,
        synthetic_data,
        strategy=args.augment_strategy
    )

    # 步骤4: 可选SMOTE
    if args.use_smote:
        print("\n应用SMOTE...")
        X_train_final, y_train_final = balance_samples(
            X_train_augmented,
            y_train_augmented
        )
    else:
        X_train_final = X_train_augmented
        y_train_final = y_train_augmented

    # 步骤5: 训练模型
    predictions, model_info = train_model(
        X_train_final,
        y_train_final,
        X_test,
        model_type=args.model_type
    )

    # 步骤6: 保存预测
    output_path = save_predictions(
        predictions,
        args.sample_path,
        args.output_name
    )

    # 完成
    print("\n" + "="*60)
    print("✓ 训练完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型类型: {model_info['type']}")
    print(f"输出文件: {output_path}")
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
