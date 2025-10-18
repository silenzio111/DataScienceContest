"""
GANmodel独立模型工具
提供模型训练和评估功能
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                            recall_score, f1_score, average_precision_score)
import warnings
warnings.filterwarnings('ignore')


def get_models():
    """获取所有可用模型"""
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100
        )
    }

    # 添加XGBoost
    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        )
    except ImportError:
        print("⚠️ XGBoost未安装，跳过")

    # 添加LightGBM
    try:
        import lightgbm as lgb
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42, n_estimators=100, objective='binary',
            n_jobs=-1, verbose=-1
        )
    except ImportError:
        print("⚠️ LightGBM未安装，跳过")

    return models


def evaluate_model(model, X_train, y_train, model_name):
    """评估单个模型"""
    print(f"\n=== 评估 {model_name} ===")

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    # 训练模型
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    # 计算AUPRC指标
    auprc = average_precision_score(y_train, y_pred_proba)
    baseline_auprc = y_train.mean()

    # 计算指标
    metrics = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'train_auc': roc_auc_score(y_train, y_pred_proba),
        'train_auprc': auprc,
        'baseline_auprc': baseline_auprc,
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred)
    }

    print(f"交叉验证 AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']*2:.4f})")
    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"训练集 AUPRC: {metrics['train_auprc']:.4f}")

    return model, metrics


def make_predictions(model, X_test):
    """生成预测"""
    return model.predict_proba(X_test)[:, 1]


def train_stacking_v2(X_train, y_train, strategy='simple_average', top_n=3, n_folds=5):
    """
    训练简化的stacking模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        strategy: 'simple_average' (简单平均), 'ridge' (岭回归元模型)
        top_n: 使用前N个最强模型
        n_folds: 交叉验证折数

    Returns:
        stacking_model: 训练好的stacking模型
        metrics: 性能指标
    """
    print(f"\n=== 训练Stacking模型 ===")
    print(f"策略: {strategy}")
    print(f"Top-N模型: {top_n}")

    # 获取基础模型
    all_models = get_models()

    # 选择top_n个模型
    model_names = list(all_models.keys())[:top_n]
    base_models = {name: all_models[name] for name in model_names}

    print(f"基础模型: {list(base_models.keys())}")

    # 训练基础模型
    trained_base_models = {}
    oof_predictions = []

    for name, model in base_models.items():
        print(f"\n训练 {name}...")

        # 交叉验证生成OOF预测
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X_train))

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]

            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            oof_preds[val_idx] = model_clone.predict_proba(X_val)[:, 1]

        oof_predictions.append(oof_preds)

        # 在全部数据上训练
        model_final = clone(model)
        model_final.fit(X_train, y_train)
        trained_base_models[name] = model_final

        # OOF性能
        oof_auc = roc_auc_score(y_train, oof_preds)
        print(f"  {name} OOF AUC: {oof_auc:.4f}")

    # 创建简单stacking类
    class SimpleStacking:
        def __init__(self, base_models, use_average=True):
            self.base_models = base_models
            self.use_average = use_average

        def predict_proba(self, X):
            predictions = []
            for model in self.base_models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)

            avg_pred = np.mean(predictions, axis=0)
            return np.column_stack([1 - avg_pred, avg_pred])

        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

    # 创建stacking模型
    stacking_model = SimpleStacking(trained_base_models)

    # 评估
    y_pred_proba = stacking_model.predict_proba(X_train)[:, 1]
    y_pred = stacking_model.predict(X_train)

    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba),
        'train_auprc': average_precision_score(y_train, y_pred_proba),
        'baseline_auprc': y_train.mean(),
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred)
    }

    print(f"\n=== Stacking模型性能 ===")
    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"训练集 AUPRC: {metrics['train_auprc']:.4f}")
    print(f"F1分数: {metrics['train_f1']:.4f}")

    return stacking_model, metrics
