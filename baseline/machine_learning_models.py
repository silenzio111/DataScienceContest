"""
机器学习模型函数式模块
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, average_precision_score,
                            precision_recall_curve, auc)
import warnings
warnings.filterwarnings('ignore')

def get_models():
    """获取所有模型"""
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42, class_weight='balanced'
        ),
        'naive_bayes': GaussianNB(),
        'knn': KNeighborsClassifier(n_jobs=-1)
    }

    # 尝试添加高级模型
    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        )
    except ImportError:
        print("XGBoost未安装，跳过")

    try:
        import lightgbm as lgb
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42, n_estimators=100, objective='binary',
            n_jobs=-1, verbose=-1
        )
    except ImportError:
        print("LightGBM未安装，跳过")

    return models

def evaluate_model(model, X_train, y_train, model_name):
    """评估单个模型"""
    print(f"\n=== 评估 {model_name} ===")

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    # 训练模型
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    # 计算AUPRC指标
    auprc = average_precision_score(y_train, y_pred_proba)

    # 计算PR曲线AUC
    precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
    pr_auc = auc(recall, precision)

    # 计算基线AUPRC（对于不平衡数据很重要）
    baseline_auprc = y_train.mean()
    auprc_improvement = (auprc / baseline_auprc - 1) * 100 if baseline_auprc > 0 else 0

    # 计算指标
    metrics = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'train_auc': roc_auc_score(y_train, y_pred_proba),
        'train_auprc': auprc,
        'train_pr_auc': pr_auc,
        'baseline_auprc': baseline_auprc,
        'auprc_improvement': auprc_improvement,
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred)
    }

    # 混淆矩阵
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"交叉验证 AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']*2:.4f})")
    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"训练集 AUPRC: {metrics['train_auprc']:.4f}")
    print(f"基线 AUPRC: {metrics['baseline_auprc']:.4f}")
    print(f"AUPRC提升: {metrics['auprc_improvement']:+.1f}%")
    print(f"准确率: {metrics['train_accuracy']:.4f}")
    print(f"精确率: {metrics['train_precision']:.4f}")
    print(f"召回率: {metrics['train_recall']:.4f}")
    print(f"F1分数: {metrics['train_f1']:.4f}")

    print(f"\n混淆矩阵:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    return model, metrics

def evaluate_all_models(X_train, y_train):
    """评估所有模型"""
    print("=== 开始评估所有模型 ===")

    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        try:
            trained_model, metrics = evaluate_model(model, X_train, y_train, name)
            results[name] = metrics
            trained_models[name] = trained_model
        except Exception as e:
            print(f"评估 {name} 时出错: {e}")

    # 创建结果表
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('cv_auc_mean', ascending=False)

    print("\n=== 模型评估结果 ===")
    print(results_df.round(4))

    print(f"\n最佳模型: {results_df.index[0]}")
    print(f"最佳 AUC: {results_df.iloc[0]['cv_auc_mean']:.4f}")

    return trained_models, results_df

def create_ensemble(best_models, X_train, y_train, top_n=3):
    """创建集成模型"""
    print(f"\n=== 创建集成模型 (Top {top_n}) ===")

    # 选择前N个最佳模型
    ensemble_models = list(best_models.values())[:top_n]

    class SimpleEnsemble:
        def __init__(self, models):
            self.models = models
            self.fitted_models = []

        def fit(self, X, y):
            self.fitted_models = []
            for model in self.models:
                model.fit(X, y)
                self.fitted_models.append(model)

        def predict_proba(self, X):
            if not self.fitted_models:
                raise ValueError("模型尚未训练")
            probas = []
            for model in self.fitted_models:
                proba = model.predict_proba(X)[:, 1]
                probas.append(proba)
            avg_proba = np.mean(probas, axis=0)
            return np.column_stack([1-avg_proba, avg_proba])

        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

    ensemble = SimpleEnsemble(ensemble_models)
    ensemble.fit(X_train, y_train)

    # 评估集成模型
    y_pred = ensemble.predict(X_train)
    y_pred_proba = ensemble.predict_proba(X_train)[:, 1]

    # 计算AUPRC指标
    auprc = average_precision_score(y_train, y_pred_proba)
    baseline_auprc = y_train.mean()
    auprc_improvement = (auprc / baseline_auprc - 1) * 100 if baseline_auprc > 0 else 0

    ensemble_metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba),
        'train_auprc': auprc,
        'baseline_auprc': baseline_auprc,
        'auprc_improvement': auprc_improvement,
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred)
    }

    print(f"集成模型性能:")
    print(f"AUC: {ensemble_metrics['train_auc']:.4f}")
    print(f"AUPRC: {ensemble_metrics['train_auprc']:.4f}")
    print(f"基线 AUPRC: {ensemble_metrics['baseline_auprc']:.4f}")
    print(f"AUPRC提升: {ensemble_metrics['auprc_improvement']:+.1f}%")
    print(f"准确率: {ensemble_metrics['train_accuracy']:.4f}")
    print(f"F1分数: {ensemble_metrics['train_f1']:.4f}")

    return ensemble, ensemble_metrics

def make_predictions(models, X_test, sample_path):
    """生成预测结果"""
    print("\n=== 生成预测结果 ===")

    # 读取提交样例
    try:
        sample = pd.read_csv(sample_path)
    except FileNotFoundError:
        sample = pd.DataFrame({
            'id': range(len(X_test)),
            'target': [0] * len(X_test)
        })

    predictions = {}

    for name, model in models.items():
        print(f"生成 {name} 预测...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        submission = sample.copy()
        submission['target'] = y_pred_proba
        submission_path = f"outputs/{name}_submission.csv"
        submission.to_csv(submission_path, index=False)

        predictions[name] = y_pred_proba
        print(f"保存到: {submission_path}")

    return predictions