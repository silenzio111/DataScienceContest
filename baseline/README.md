# 信贷风控 Baseline 代码库

数智风控：面向新产品的信用风险评估

版本: 2.0 (已整理优化)
更新时间: 2025-10-18

---

## 📁 项目结构

```
baseline/
├── 核心模块
│   ├── data_preprocessing.py          数据预处理模块
│   ├── machine_learning_models.py     基础机器学习模型
│   ├── stacking_models.py             Stacking集成模型
│   └── plot_predictions.py            可视化分析工具
│
├── 运行脚本
│   ├── credit_risk_main.py            主入口 (Baseline流程)
│   ├── run_stacking.py                Stacking训练脚本
│   └── generate_predictions.py        预测生成工具
│
├── 输出目录
│   ├── outputs/                       所有输出结果
│   │   ├── *_submission.csv          预测提交文件
│   │   ├── *_report.md               训练报告
│   │   └── plots/                    可视化图表
│
└── 文档
    ├── README.md                      本文件
    ├── CODE_STRUCTURE.md              代码架构文档
    └── STACKING_README.md             Stacking使用指南
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 必需包
pandas numpy scikit-learn matplotlib seaborn openpyxl imbalanced-learn

# 可选包 (强烈推荐)
xgboost lightgbm
```

### 2. 运行Baseline

```bash
cd baseline

# 运行完整baseline流程
python credit_risk_main.py

# 输出: outputs/ensemble_submission.csv (最佳baseline, 得分0.5607)
```

### 3. 运行Stacking模型 ⭐ 推荐

```bash
# 使用简单平均策略 (推荐首选)
python run_stacking.py --strategy simple_average

# 使用Ridge正则化策略
python run_stacking.py --strategy ridge --C 0.1

# 测试所有策略
python run_stacking.py --test-all

# 输出: outputs/stacking_*_submission.csv
```

### 4. 生成特定模型预测

```bash
# 查看可用模型
python generate_predictions.py --list

# 生成XGBoost预测
python generate_predictions.py --models xgboost

# 生成多个模型预测
python generate_predictions.py --models knn decision_tree naive_bayes

# 生成所有模型预测
python generate_predictions.py --all
```

### 5. 可视化分析

```bash
# 生成所有预测文件的可视化分析
python plot_predictions.py

# 输出: outputs/plots/ 目录下的图表和汇总
```

---

## 🎯 推荐使用方案

基于实际比赛反馈（Ensemble: 0.5607 > Stacking v1: 0.5426），推荐策略：

### 首选方案

```bash
python run_stacking.py --strategy simple_average --output-name stacking_best
```

**原因**:
- 完全模仿Ensemble成功策略
- Top3模型（XGBoost + Random Forest + Gradient Boosting）简单平均
- 预测分布最合理
- 预期得分: 0.55-0.58

### 备选方案

```bash
python run_stacking.py --strategy ridge --C 0.1 --output-name stacking_ridge
```

**原因**:
- 中等正则化避免过拟合
- 元模型智能学习组合权重
- 预测均值接近Ensemble
- 预期得分: 0.54-0.57

---

## 📊 模型性能对比

### 已验证模型

| 模型 | 预测均值 | CV AUC | 实际得分 | 推荐度 |
|------|----------|--------|----------|--------|
| **ensemble** | 0.0565 | - | **0.5607** | 🏆 |
| stacking_simple_avg | 0.0419 | ~0.999 | ? | ⭐⭐⭐ |
| stacking_ridge | 0.0531 | ~0.999 | ? | ⭐⭐ |
| stacking v1 | 0.0103 | 0.9992 | 0.5426 | ❌ |
| gradient_boosting | 0.0424 | 0.9984 | ? | ✅ |
| xgboost | 0.0163 | 0.9989 | ? | ⚠️ |
| random_forest | 0.0670 | 0.9990 | ? | ✅ |

---

## 📝 核心模块说明

### 1. data_preprocessing.py

**功能**: 完整的数据预处理流水线

**主要函数**:
- `preprocess_pipeline()` - 主流水线函数
- `load_data()` - 加载Excel数据
- `handle_missing_values()` - 处理缺失值
- `create_features()` - 特征工程
- `balance_samples()` - SMOTE样本平衡

**使用**:
```python
from data_preprocessing import preprocess_pipeline

X_train, X_test, y_train = preprocess_pipeline(
    train_path="../初赛选手数据/训练数据集.xlsx",
    test_path="../初赛选手数据/测试集.xlsx"
)
```

### 2. machine_learning_models.py

**功能**: 基础机器学习模型训练和评估

**主要函数**:
- `get_models()` - 获取所有可用模型
- `evaluate_model()` - 评估单个模型
- `evaluate_all_models()` - 评估所有模型
- `create_ensemble()` - 创建集成模型
- `make_predictions()` - 生成预测

**支持模型**:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM (可选)
- Decision Tree
- Naive Bayes
- KNN

### 3. stacking_models.py

**功能**: 高级Stacking集成模型

**主要类和函数**:
- `OptimizedStackingClassifier` - Stacking分类器类
- `train_stacking_v2()` - 训练Stacking模型
- `get_base_models()` - 获取基础模型
- `get_meta_model()` - 获取元模型

**支持策略**:
- `simple_average` - 简单平均 (推荐)
- `ridge` - Ridge正则化元模型
- `xgboost` - XGBoost元模型

### 4. plot_predictions.py

**功能**: 预测结果可视化分析

**生成内容**:
- 每个模型的预测分布图
- 模型对比图
- 统计汇总表
- 风险分布分析

---

## ⚙️ 高级配置

### Stacking参数说明

```bash
python run_stacking.py \
    --strategy simple_average \  # 策略选择
    --top-n 3 \                  # 使用Top N个模型
    --n-folds 5 \                # 交叉验证折数
    --C 0.1 \                    # 正则化强度 (仅Ridge)
    --output-name my_stacking    # 输出文件名
```

**参数详解**:
- `--strategy`:
  - `simple_average`: Top N模型简单平均
  - `ridge`: Ridge正则化元模型
  - `xgboost`: XGBoost元模型
- `--top-n`: 使用前N个最强模型 (推荐3)
- `--n-folds`: K折交叉验证 (推荐5)
- `--C`: 正则化强度，越小正则化越强 (推荐0.1)

---

## 🔍 常见问题

### Q1: 为什么Ensemble比Stacking v1好？

**A**: Stacking v1预测过于保守（均值0.0103），而Ensemble预测更合理（0.0565）。新版Stacking已优化。

### Q2: 应该使用哪个Stacking策略？

**A**: 推荐 `simple_average`，因为：
- 模仿Ensemble成功策略
- 避免元模型过拟合
- 预测分布合理

### Q3: 如何添加新模型？

**A**: 在 `machine_learning_models.py` 的 `get_models()` 函数中添加：
```python
models['my_model'] = MyModelClass(params...)
```

### Q4: 训练集AUC=1.0是否过拟合？

**A**: 可能过拟合。关注：
- 交叉验证AUC (更重要)
- 预测分布是否合理
- 与Ensemble对比

---

## 📈 性能优化建议

### 1. 避免过拟合
- ✅ 使用简单平均而非复杂元模型
- ✅ 添加正则化 (C=0.1)
- ✅ 关注预测分布而非训练指标

### 2. 提升泛化能力
- ✅ 只使用Top3最强模型
- ✅ 5折交叉验证
- ✅ SMOTE样本平衡

### 3. 预测分布校准
- ✅ 确保预测均值接近Ensemble (0.05-0.06)
- ✅ 避免过于保守 (<0.02) 或激进 (>0.10)

---

## 📚 相关文档

- **`CODE_STRUCTURE.md`**: 代码架构详细说明
- **`STACKING_README.md`**: Stacking模型完整指南
- **`outputs/STACKING_OPTIMIZATION_ANALYSIS.md`**: 优化分析报告
- **`outputs/FINAL_MODEL_COMPARISON.md`**: 模型对比分析

---

## 🛠️ 维护日志

### v2.0 (2025-10-18) - 代码整理
- ✅ 合并Stacking模型文件 (v1+v2 → v1)
- ✅ 合并运行脚本 (统一接口)
- ✅ 创建通用预测生成工具
- ✅ 完善文档和使用说明
- ✅ 删除冗余Voting模型
- ✅ 优化代码结构

### v1.0 (2025-10-17) - 初始版本
- 基础Baseline实现
- Stacking集成模型
- 可视化工具
- 多个预测文件生成

---

## 📧 反馈与支持

如有问题或建议，请查看：
- 代码注释和文档字符串
- outputs/目录下的各种报告
- 相关技术文档

---

**Happy Coding! 祝比赛顺利！** 🎯
