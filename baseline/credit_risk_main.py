"""
信贷风控Baseline - 主运行脚本
函数式编程实现
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# 导入自定义模块
from data_preprocessing import preprocess_pipeline
from machine_learning_models import evaluate_all_models, create_ensemble, make_predictions

def main():
    """主函数"""
    print("=== 信贷风控Baseline解决方案 ===")
    print("开始时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()

    # 1. 数据预处理
    print("第1步: 数据预处理")
    train_path = "../初赛选手数据/训练数据集.xlsx"
    test_path = "../初赛选手数据/测试集.xlsx"

    try:
        X_train, X_test, y_train = preprocess_pipeline(train_path, test_path)
        print("✅ 数据预处理完成")
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return

    # 2. 模型训练和评估
    print("\n第2步: 模型训练和评估")
    try:
        trained_models, results_df = evaluate_all_models(X_train, y_train)
        print("✅ 模型训练完成")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return

    # 3. 创建集成模型
    print("\n第3步: 创建集成模型")
    try:
        ensemble, ensemble_metrics = create_ensemble(trained_models, X_train, y_train, top_n=3)
        print("✅ 集成模型创建完成")
    except Exception as e:
        print(f"❌ 集成模型创建失败: {e}")
        return

    # 4. 生成预测结果
    print("\n第4步: 生成预测结果")
    try:
        sample_path = "../初赛选手数据/提交样例.csv"
        best_models = list(trained_models.values())[:3]
        best_model_names = list(trained_models.keys())[:3]

        # 将集成模型也加入预测
        prediction_models = {
            'ensemble': ensemble
        }
        for name, model in zip(best_model_names, best_models):
            prediction_models[name] = model

        predictions = make_predictions(prediction_models, X_test, sample_path)
        print("✅ 预测结果生成完成")
    except Exception as e:
        print(f"❌ 预测结果生成失败: {e}")
        return

    # 5. 保存结果和报告
    print("\n第5步: 保存结果")
    try:
        # 保存评估结果
        results_df.to_csv('outputs/model_results.csv', index=False)

        # 创建简单的报告
        report = f"""# 信贷风控Baseline训练报告

## 训练时间
- 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 训练耗时: 约几分钟

## 数据信息
- 训练集样本数: {len(X_train)}
- 测试集样本数: {len(X_test)}
- 特征数量: {X_train.shape[1]}

## 模型性能排名
"""
        for i, (idx, row) in enumerate(results_df.iterrows(), 1):
            report += f"""
{i}. **{idx}**
   - 交叉验证AUC: {row['cv_auc_mean']:.4f}
   - 训练集AUC: {row['train_auc']:.4f}
   - 训练集AUPRC: {row['train_auprc']:.4f}
   - 基线AUPRC: {row['baseline_auprc']:.4f}
   - AUPRC提升: {row['auprc_improvement']:+.1f}%
   - 准确率: {row['train_accuracy']:.4f}
   - F1分数: {row['train_f1']:.4f}
"""

        report += f"""
## 集成模型性能
- AUC: {ensemble_metrics['train_auc']:.4f}
- AUPRC: {ensemble_metrics['train_auprc']:.4f}
- 基线AUPRC: {ensemble_metrics['baseline_auprc']:.4f}
- AUPRC提升: {ensemble_metrics['auprc_improvement']:+.1f}%
- 准确率: {ensemble_metrics['train_accuracy']:.4f}
- F1分数: {ensemble_metrics['train_f1']:.4f}

## 输出文件
- 模型结果: outputs/model_results.csv
- 预测文件: outputs/*_submission.csv

## AUPRC指标说明
- **AUPRC** (Average Precision-Recall Curve Area): 衡量模型在不平衡数据上的性能
- **基线AUPRC**: 随机分类器的性能，等于正样本比例
- **AUPRC提升**: 相对于随机分类器的性能提升百分比
- 对于不平衡数据集，AUPRC比AUC更能反映模型的真实性能

## 建议
1. 使用集成模型进行最终预测
2. 重点关注AUPRC指标，特别是对于不平衡数据集
3. 关注预测概率较高的客户
4. 定期重新训练模型
"""

        with open('outputs/training_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("✅ 结果保存完成")
    except Exception as e:
        print(f"❌ 结果保存失败: {e}")

    # 6. 完成
    print("\n" + "="*50)
    print("🎉 Baseline训练完成!")
    print(f"最佳模型: {results_df.index[0]}")
    print(f"最佳AUC: {results_df.iloc[0]['cv_auc_mean']:.4f}")
    print(f"集成模型AUC: {ensemble_metrics['train_auc']:.4f}")
    print(f"结果保存在: outputs/ 目录")
    print("="*50)

if __name__ == "__main__":
    # 确保outputs目录存在
    os.makedirs('outputs', exist_ok=True)

    # 运行主程序
    main()