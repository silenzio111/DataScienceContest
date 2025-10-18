"""
Stacking集成模型统一运行脚本
支持多种stacking策略，参数化配置
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# 导入必要模块
from data_preprocessing import preprocess_pipeline
from stacking_models import train_stacking_v2, make_stacking_v2_predictions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stacking集成模型训练')

    parser.add_argument('--strategy', type=str, default='simple_average',
                       choices=['simple_average', 'ridge', 'xgboost'],
                       help='Stacking策略: simple_average (简单平均), ridge (Ridge正则化), xgboost (XGBoost元模型)')

    parser.add_argument('--top-n', type=int, default=3,
                       help='使用Top N个最强模型 (默认: 3)')

    parser.add_argument('--n-folds', type=int, default=5,
                       help='交叉验证折数 (默认: 5)')

    parser.add_argument('--C', type=float, default=0.1,
                       help='正则化强度 (仅Ridge策略, 默认: 0.1, 越小正则化越强)')

    parser.add_argument('--output-name', type=str, default='stacking',
                       help='输出文件名前缀 (默认: stacking)')

    parser.add_argument('--test-all', action='store_true',
                       help='测试所有策略组合')

    return parser.parse_args()


def run_single_strategy(X_train, y_train, X_test, sample_path,
                       strategy='simple_average', top_n=3, n_folds=5, C=0.1, output_name='stacking'):
    """运行单个stacking策略"""

    print("\n" + "="*60)
    print(f"策略: {strategy} | Top-{top_n} | C={C}")
    print("="*60)

    # 训练模型
    model, metrics = train_stacking_v2(
        X_train, y_train,
        strategy=strategy,
        top_n=top_n,
        n_folds=n_folds,
        C=C
    )

    # 生成预测
    predictions = make_stacking_v2_predictions(
        model, X_test, sample_path,
        output_name=output_name
    )

    return {
        'strategy': strategy,
        'top_n': top_n,
        'C': C,
        'output_name': output_name,
        'metrics': metrics,
        'pred_mean': predictions.mean(),
        'pred_std': predictions.std()
    }


def test_all_strategies(X_train, y_train, X_test, sample_path):
    """测试所有推荐的策略组合"""

    strategies = [
        {
            'strategy': 'simple_average',
            'top_n': 3,
            'C': 0.1,  # 不使用，但需要提供
            'output_name': 'stacking_simple_avg',
            'desc': 'Top3简单平均（推荐首选）'
        },
        {
            'strategy': 'ridge',
            'top_n': 3,
            'C': 0.1,
            'output_name': 'stacking_ridge_medium',
            'desc': 'Top3 + 中等正则化'
        },
        {
            'strategy': 'ridge',
            'top_n': 3,
            'C': 0.01,
            'output_name': 'stacking_ridge_strong',
            'desc': 'Top3 + 强正则化'
        },
        {
            'strategy': 'ridge',
            'top_n': 3,
            'C': 1.0,
            'output_name': 'stacking_ridge_weak',
            'desc': 'Top3 + 弱正则化'
        },
    ]

    results = []

    for idx, config in enumerate(strategies, 1):
        print("\n" + "="*60)
        print(f"测试 {idx}/{len(strategies)}: {config['desc']}")
        print("="*60)

        result = run_single_strategy(
            X_train, y_train, X_test, sample_path,
            strategy=config['strategy'],
            top_n=config['top_n'],
            n_folds=5,
            C=config['C'],
            output_name=config['output_name']
        )
        results.append(result)

    # 生成汇总报告
    print("\n" + "="*60)
    print("所有策略测试完成 - 汇总")
    print("="*60)

    summary_data = []
    for r in results:
        summary_data.append({
            '策略': r['strategy'],
            'Top-N': r['top_n'],
            'C': r['C'],
            '输出文件': f"{r['output_name']}_submission.csv",
            'AUC': f"{r['metrics']['train_auc']:.4f}",
            'F1': f"{r['metrics']['train_f1']:.4f}",
            '预测均值': f"{r['pred_mean']:.4f}",
            '预测标准差': f"{r['pred_std']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # 保存汇总
    summary_df.to_csv('outputs/stacking_strategies_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 汇总已保存到: outputs/stacking_strategies_summary.csv")

    return results


def main():
    """主函数"""
    args = parse_args()

    print("="*60)
    print("    Stacking集成模型训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 配置路径
    train_path = "../初赛选手数据/训练数据集.xlsx"
    test_path = "../初赛选手数据/测试集.xlsx"
    sample_path = "../初赛选手数据/提交样例.csv"

    # 检查文件
    for path, name in [(train_path, "训练集"), (test_path, "测试集"), (sample_path, "提交样例")]:
        if not os.path.exists(path):
            print(f"❌ 错误: 找不到{name}文件: {path}")
            return 1

    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)

    # 数据预处理
    print("\n" + "="*60)
    print("步骤 1: 数据预处理")
    print("="*60)

    X_train, X_test, y_train = preprocess_pipeline(train_path, test_path)

    if X_train is None:
        print("❌ 数据预处理失败")
        return 1

    # 训练模型
    print("\n" + "="*60)
    print("步骤 2: 训练Stacking模型")
    print("="*60)

    if args.test_all:
        # 测试所有策略
        results = test_all_strategies(X_train, y_train, X_test, sample_path)
    else:
        # 运行单个策略
        result = run_single_strategy(
            X_train, y_train, X_test, sample_path,
            strategy=args.strategy,
            top_n=args.top_n,
            n_folds=args.n_folds,
            C=args.C,
            output_name=args.output_name
        )

        print("\n" + "="*60)
        print("训练完成")
        print("="*60)
        print(f"策略: {result['strategy']}")
        print(f"输出文件: outputs/{result['output_name']}_submission.csv")
        print(f"训练AUC: {result['metrics']['train_auc']:.4f}")
        print(f"训练F1: {result['metrics']['train_f1']:.4f}")
        print(f"预测均值: {result['pred_mean']:.4f}")
        print(f"预测标准差: {result['pred_std']:.4f}")

    # 完成
    print("\n" + "="*60)
    print("✓ 全部任务完成!")
    print("="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
