# Stacking集成模型训练报告

## 训练时间
- 时间: 2025-10-17 23:49:12

## 模型架构
- **第一层（Base Models）**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - KNN

- **第二层（Meta Model）**:
  - Logistic Regression / XGBoost

## 模型性能

- **训练集 AUC**: 1.0000
- **训练集 AUPRC**: 1.0000
- **基线 AUPRC**: 0.5000
- **AUPRC提升**: +100.0%
- **准确率**: 1.0000
- **精确率**: 1.0000
- **召回率**: 1.0000
- **F1分数**: 1.0000

## Stacking原理

Stacking（堆叠）是一种集成学习方法：

1. **第一层**: 训练多个不同的基础模型
2. **第二层**: 使用元模型基于基础模型的预测进行最终预测
3. **关键技术**: 使用out-of-fold预测避免过拟合

## 优势

- 结合多个模型的优点
- 比简单平均更智能
- 可以学习不同模型的权重和组合方式
- 通常比单一模型性能更好

## 输出文件

- 预测文件: `outputs/stacking_submission.csv`
- 训练报告: `outputs/stacking_report.md`

---
生成时间: 2025-10-17 23:49:12
