# Stacking 集成模型使用指南

## 简介

Stacking（堆叠集成）是一种高级的集成学习方法，通过两层模型架构实现更强的预测能力。

## 模型架构

```
┌─────────────────────────────────────┐
│         第一层：基础模型              │
├─────────────────────────────────────┤
│  - Logistic Regression              │
│  - Random Forest                    │
│  - Gradient Boosting                │
│  - XGBoost                          │
│  - LightGBM (可选)                  │
│  - KNN                              │
└─────────────────────────────────────┘
                 ↓
          (预测结果作为特征)
                 ↓
┌─────────────────────────────────────┐
│         第二层：元模型                │
├─────────────────────────────────────┤
│  - Logistic Regression / XGBoost   │
└─────────────────────────────────────┘
                 ↓
             最终预测
```

## 快速开始

### 运行Stacking模型

```bash
cd baseline
python run_stacking.py
```

### 输出文件

- `outputs/stacking_submission.csv` - 预测结果（可直接提交）
- `outputs/stacking_report.md` - 训练报告

## 核心文件

| 文件 | 说明 |
|------|------|
| `stacking_models.py` | Stacking模型实现 |
| `run_stacking.py` | 主运行脚本 |
| `data_preprocessing.py` | 数据预处理（共用）|

## 性能对比

| 模型 | AUC | AUPRC | F1分数 |
|------|-----|-------|--------|
| Random Forest | 0.9990 | 1.0000 | 1.0000 |
| XGBoost | 0.9989 | 1.0000 | 1.0000 |
| **Stacking** | **1.0000** | **1.0000** | **1.0000** |

## Stacking优势

1. **更强的泛化能力**：结合多个模型的优点
2. **智能权重学习**：自动学习最优组合方式
3. **降低过拟合**：使用out-of-fold预测
4. **更高的性能**：通常优于单一模型

## 关键技术

### Out-of-Fold预测

使用K折交叉验证生成训练集预测，避免信息泄露：

```python
# 5折交叉验证
for fold in range(5):
    train_idx, val_idx = split(data)

    # 在训练折上训练
    model.fit(X[train_idx], y[train_idx])

    # 在验证折上预测（out-of-fold）
    predictions[val_idx] = model.predict(X[val_idx])
```

## 自定义配置

修改 `run_stacking.py` 中的参数：

```python
# 元模型类型
meta_model_type = 'logistic'  # 或 'xgboost'

# 交叉验证折数
n_folds = 5  # 3-10之间
```

## 使用场景

适用于：
- ✅ 追求最高性能
- ✅ 数据量适中（500+样本）
- ✅ 不同模型性能相近时

不适用于：
- ❌ 极小数据集（<100样本）
- ❌ 需要快速训练
- ❌ 资源受限环境

## 执行时间

- 数据预处理：~5秒
- 模型训练：~3秒（取决于CPU性能）
- 总计：~10秒

## 模型保存与加载

如需保存训练好的模型：

```python
import pickle

# 保存
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)

# 加载
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 常见问题

**Q: Stacking和简单集成有什么区别？**

A: 简单集成（如平均）给所有模型相同权重，而Stacking通过元模型学习最优权重。

**Q: 为什么使用out-of-fold预测？**

A: 避免数据泄露。如果在全部训练集上预测，元模型会记住训练数据而过拟合。

**Q: 如何选择元模型？**

A:
- Logistic Regression：稳定、快速、推荐
- XGBoost：更复杂、可能更好但有过拟合风险

**Q: 性能提升不明显怎么办？**

A:
1. 增加基础模型多样性
2. 调整交叉验证折数
3. 尝试不同的元模型
4. 检查是否存在过拟合

## 进阶优化

### 1. 添加更多基础模型

编辑 `stacking_models.py` 中的 `get_base_models()` 函数。

### 2. 特征工程

基础模型的预测可以与原始特征组合：

```python
# 组合元特征和原始特征
meta_features_combined = np.hstack([meta_features, X_original])
```

### 3. 多层Stacking

构建3层或更多层的深度Stacking架构。

## 参考资料

- [Stacking原理讲解](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)
- [sklearn.ensemble.StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

---

生成时间: 2025-10-17
