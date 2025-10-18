"""
通用预测生成工具
支持为任意模型生成预测文件
"""

import os
import argparse
import pandas as pd
from data_preprocessing import preprocess_pipeline
from machine_learning_models import get_models, evaluate_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为指定模型生成预测文件')

    parser.add_argument('--models', type=str, nargs='+',
                       help='要生成预测的模型名称 (如: xgboost knn decision_tree)')

    parser.add_argument('--all', action='store_true',
                       help='为所有可用模型生成预测')

    parser.add_argument('--list', action='store_true',
                       help='列出所有可用模型')

    return parser.parse_args()


def list_available_models():
    """列出所有可用模型"""
    models = get_models()

    print("\n可用模型列表:")
    print("="*60)
    for idx, name in enumerate(models.keys(), 1):
        print(f"{idx:2d}. {name}")
    print("="*60)
    print(f"总计: {len(models)} 个模型")
    print()


def generate_prediction(model_name, X_train, y_train, X_test, sample_path):
    """为单个模型生成预测"""

    all_models = get_models()

    if model_name not in all_models:
        print(f"❌ 模型 '{model_name}' 不存在")
        return False

    print(f"\n{'='*60}")
    print(f"处理模型: {model_name}")
    print('='*60)

    try:
        # 训练模型
        model = all_models[model_name]
        trained_model, metrics = evaluate_model(model, X_train, y_train, model_name)

        # 生成预测
        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]

        # 读取提交样例
        try:
            sample = pd.read_csv(sample_path)
        except FileNotFoundError:
            sample = pd.DataFrame({
                'id': range(len(X_test)),
                'target': [0] * len(X_test)
            })

        # 创建提交文件
        submission = sample.copy()
        submission['target'] = y_pred_proba

        output_path = f"outputs/{model_name}_submission.csv"
        submission.to_csv(output_path, index=False)

        print(f"✓ 预测已保存: {output_path}")
        print(f"  统计: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

        return True

    except Exception as e:
        print(f"❌ 生成预测失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()

    # 如果要列出模型
    if args.list:
        list_available_models()
        return 0

    # 如果既没有指定模型也没有指定--all
    if not args.models and not args.all:
        print("❌ 错误: 请指定要生成预测的模型")
        print("使用 --models <模型名> 指定具体模型")
        print("使用 --all 为所有模型生成预测")
        print("使用 --list 查看可用模型")
        return 1

    print("="*60)
    print("    通用预测生成工具")
    print("="*60)
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
    print("数据预处理...")
    X_train, X_test, y_train = preprocess_pipeline(train_path, test_path)

    if X_train is None:
        print("❌ 数据预处理失败")
        return 1

    # 确定要生成预测的模型
    if args.all:
        all_models = get_models()
        model_names = list(all_models.keys())
        print(f"\n为所有 {len(model_names)} 个模型生成预测...")
    else:
        model_names = args.models

    # 生成预测
    success_count = 0
    fail_count = 0

    for model_name in model_names:
        success = generate_prediction(model_name, X_train, y_train, X_test, sample_path)
        if success:
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    print("\n" + "="*60)
    print("完成")
    print("="*60)
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"输出目录: outputs/")
    print()

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    try:
        import sys
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
