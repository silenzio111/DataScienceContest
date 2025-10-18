# GANmodel - 数据增强与生成

用于信贷风控竞赛的GAN模型数据增强

创建时间: 2025-10-18

---

## 📁 目录说明

本目录用于GAN（生成对抗网络）相关的数据增强实验。

### 目的

- 解决训练集样本不足问题（仅500条）
- 解决正负样本严重不平衡问题（违约率2%）
- 生成高质量的合成数据增强训练集

---

## 🎯 应用场景

### 当前挑战

1. **样本量少**: 训练集仅500条
2. **极度不平衡**: 违约样本仅10条
3. **分布偏差**: 训练集与测试集可能存在分布差异

### GAN的潜在价值

- ✅ 生成更多违约样本
- ✅ 学习真实数据分布
- ✅ 提升模型泛化能力
- ✅ 缓解过拟合问题

---

## 📋 待实现功能

### 1. 基础GAN模型
- [x] Vanilla GAN (via CTGAN)
- [x] WGAN (via CTGAN内部实现)
- [ ] WGAN-GP (带梯度惩罚)

### 2. 条件GAN
- [x] CGAN (Conditional GAN) - CTGAN实现
- [ ] ACGAN (Auxiliary Classifier GAN)
- [x] 用于生成特定类别（违约/正常）样本

### 3. 表格数据专用GAN
- [x] CTGAN (Conditional Tabular GAN) - 已实现
- [ ] TVAE (Tabular VAE)
- [ ] TableGAN

### 4. 数据增强策略
- [x] 只增强少数类（违约样本）
- [x] 平衡增强
- [x] 混合真实+合成数据

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd GANmodel
pip install -r requirements.txt
```

### 2. 训练GAN模型

```bash
# 使用SDV的CTGAN（推荐）
python train_gan.py --use-sdv --target-class minority --epochs 300

# 参数说明:
#   --target-class: both(全部), minority(只违约), majority(只正常)
#   --epochs: 训练轮数 (默认300)
#   --batch-size: 批次大小 (默认500)
```

### 3. 生成合成数据

```bash
python generate_synthetic_data.py \
    --use-sdv \
    --target-class minority \
    --num-samples 500 \
    --output-name synthetic_minority
```

### 4. 评估合成数据质量

```bash
python evaluate_synthetic_data.py \
    --real-data ../初赛选手数据/训练数据集.xlsx \
    --synthetic-data synthetic_data/synthetic_minority.csv
```

### 5. 使用GAN增强数据训练模型（G-XGBoost）

```bash
# 方法1: 使用预生成的合成数据
python train_with_gan.py \
    --synthetic-data-path synthetic_data/synthetic_minority.csv \
    --augment-strategy minority \
    --model-type stacking \
    --output-name g_stacking

# 方法2: 直接使用GAN模型生成
python train_with_gan.py \
    --gan-model-path models/sdv_ctgan_minority_latest.pkl \
    --num-synthetic 500 \
    --use-sdv \
    --augment-strategy minority \
    --model-type stacking \
    --output-name g_stacking
```

---

## 📊 预期收益

### 数据层面
- 训练集扩大: 500 → 1000+ 条
- 违约样本: 10 → 200+ 条
- 分布更均衡

### 模型层面
- 减少过拟合
- 提升泛化能力
- 可能提升测试得分

---

## ⚠️ 注意事项

### 风险

1. **质量控制**: 合成数据质量需要严格验证
2. **分布偏移**: 可能引入不真实的模式
3. **过度拟合**: 模型可能学习GAN的伪影
4. **计算成本**: GAN训练需要时间和资源

### 最佳实践

- ✅ 先评估合成数据质量再使用
- ✅ 对比使用/不使用GAN的模型性能
- ✅ 保持一定比例的真实数据
- ✅ 使用交叉验证评估
- ✅ 监控分布统计指标

---

## 📚 参考资料

### 论文
- GAN: Goodfellow et al., 2014
- WGAN: Arjovsky et al., 2017
- CTGAN: Xu et al., 2019

### 工具库
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [ydata-synthetic](https://github.com/ydataai/ydata-synthetic)

---

## 🔄 与Baseline的集成

```python
# 1. 生成合成数据
synthetic_data = gan.generate(n_samples=500)

# 2. 合并真实数据和合成数据
X_train_augmented = pd.concat([X_train_real, synthetic_data])

# 3. 使用增强后的数据训练模型
model.fit(X_train_augmented, y_train_augmented)
```

---

## 📝 待办事项

- [x] 研究适合表格数据的GAN架构
- [x] 实现基础CTGAN模型
- [x] 设计数据质量评估指标
- [x] 实验不同的增强策略
- [x] 与baseline模型集成测试
- [ ] 分析对最终得分的影响

---

## 📂 文件结构

```
GANmodel/
├── train_gan.py                    # GAN模型训练脚本
├── generate_synthetic_data.py      # 合成数据生成脚本
├── evaluate_synthetic_data.py      # 数据质量评估脚本
├── train_with_gan.py               # G-XGBoost训练脚本
├── requirements.txt                # Python依赖
├── README.md                       # 本文件
│
├── models/                         # GAN模型保存目录
│   ├── sdv_ctgan_minority_latest.pkl
│   ├── sdv_ctgan_both_latest.pkl
│   └── test_samples/
│
├── synthetic_data/                 # 合成数据保存目录
│   ├── synthetic_minority.csv
│   └── synthetic_both.csv
│
└── evaluation/                     # 评估结果保存目录
    ├── evaluation_report.md
    ├── feature_distributions.png
    ├── correlation_heatmaps.png
    └── pca_comparison.png
```

---

## 🎯 G-XGBoost方法说明

基于论文《A Credit Risk Model with Small Sample Data Based on G-XGBoost》

### 核心思想

1. **问题**: 小样本（500条）+ 极度不平衡（违约率2%）导致模型性能差
2. **解决方案**: 使用GAN生成高质量合成数据扩充训练集
3. **优势**:
   - 生成数据分布与真实数据一致
   - 显著提升模型区分能力（KS值）
   - 略微提升预测准确率（AUC值）

### 实现流程

```
1. 训练GAN模型
   ↓
2. 生成合成数据
   ↓
3. 质量评估
   ↓
4. 数据增强（真实+合成）
   ↓
5. 训练XGBoost/Stacking
   ↓
6. 预测提交
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--target-class` | `minority` | 只生成违约样本效果最好 |
| `--epochs` | `300` | 根据数据量调整，样本越少需要越多轮 |
| `--num-synthetic` | `500-1000` | 论文建议扩充到原始2-3倍 |
| `--augment-strategy` | `minority` | 只增强违约样本 |
| `--model-type` | `stacking` | Stacking效果优于单模型 |

---

## 📊 实验设计建议

### 实验1: 基准对比

```bash
# 不使用GAN（baseline）
cd ../baseline
python run_stacking.py --strategy simple_average

# 使用GAN（G-XGBoost）
cd ../GANmodel
python train_with_gan.py \
    --synthetic-data-path synthetic_data/synthetic_minority.csv \
    --augment-strategy minority \
    --model-type stacking \
    --output-name g_stacking_500
```

### 实验2: 样本数量影响

测试不同合成样本数量（100, 200, 500, 1000, 2000）对模型性能的影响

```bash
for num in 100 200 500 1000 2000; do
    python generate_synthetic_data.py \
        --use-sdv \
        --target-class minority \
        --num-samples $num \
        --output-name synthetic_minority_$num

    python train_with_gan.py \
        --synthetic-data-path synthetic_data/synthetic_minority_$num.csv \
        --augment-strategy minority \
        --model-type stacking \
        --output-name g_stacking_$num
done
```

### 实验3: 增强策略对比

测试不同增强策略（minority, both, balanced）

```bash
for strategy in minority both balanced; do
    python train_with_gan.py \
        --gan-model-path models/sdv_ctgan_${strategy}_latest.pkl \
        --num-synthetic 500 \
        --use-sdv \
        --augment-strategy $strategy \
        --model-type stacking \
        --output-name g_stacking_$strategy
done
```

---

## 🎓 预期目标

如果GAN增强效果好：
- 基准得分: 0.5607 (ensemble)
- 目标得分: 0.57 - 0.59 (+1-3%)

---

**状态**: ✅ 已实现，可以使用

**实现内容**:
- ✅ CTGAN模型训练
- ✅ 合成数据生成
- ✅ 数据质量评估
- ✅ G-XGBoost集成训练
- ✅ 多种增强策略
- ✅ 完整工作流程

---

更新时间: 2025-10-18
