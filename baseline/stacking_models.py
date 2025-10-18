"""
优化的Stacking集成模型模块
基于比赛反馈优化，提供多种stacking策略
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                            recall_score, f1_score, average_precision_score)
import warnings
warnings.filterwarnings('ignore')


def get_base_models():
    """获取第一层基础模型"""
    models = {
        'lr': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'rf': RandomForestClassifier(
            random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            random_state=42, n_estimators=100
        ),
        'knn': KNeighborsClassifier(n_jobs=-1)
    }

    # 添加高级模型
    try:
        import xgboost as xgb
        models['xgb'] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        )
    except ImportError:
        print("⚠️ XGBoost未安装，跳过")

    try:
        import lightgbm as lgb
        models['lgb'] = lgb.LGBMClassifier(
            random_state=42, n_estimators=100, objective='binary',
            n_jobs=-1, verbose=-1
        )
    except ImportError:
        print("⚠️ LightGBM未安装，跳过")

    return models


def get_top_base_models(top_n=3):
    """只获取Top N个最强的基础模型"""
    models = {}

    # 按照已知性能排序，只选择最好的
    try:
        import xgboost as xgb
        models['xgb'] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        )
    except ImportError:
        pass

    models['rf'] = RandomForestClassifier(
        random_state=42, n_estimators=100, class_weight='balanced', n_jobs=-1
    )

    models['gb'] = GradientBoostingClassifier(
        random_state=42, n_estimators=100
    )

    # 只返回前top_n个
    return dict(list(models.items())[:top_n])


def get_meta_model(model_type='ridge', C=1.0):
    """获取第二层元模型（带正则化选项）"""
    if model_type == 'ridge':
        # Ridge回归版本的逻辑回归，更强的正则化
        return LogisticRegression(
            random_state=42, max_iter=1000,
            penalty='l2', C=C, solver='lbfgs'
        )
    elif model_type == 'logistic':
        return LogisticRegression(random_state=42, max_iter=1000, C=C)
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=42, n_estimators=50, eval_metric='logloss',
                use_label_encoder=False, max_depth=3,
                learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0
            )
        except ImportError:
            print("⚠️ XGBoost未安装，使用Ridge Logistic")
            return LogisticRegression(
                random_state=42, max_iter=1000,
                penalty='l2', C=0.1, solver='lbfgs'
            )
    else:
        return LogisticRegression(random_state=42, max_iter=1000)


class OptimizedStackingClassifier:
    """
    优化的Stacking集成分类器

    改进点：
    1. 支持更强的正则化避免过拟合
    2. 支持只使用Top模型
    3. 支持简单平均作为baseline
    """

    def __init__(self, base_models, meta_model, n_folds=5, use_simple_average=False):
        """
        参数:
            base_models: dict, 基础模型字典
            meta_model: 元模型
            n_folds: int, 交叉验证折数
            use_simple_average: bool, 是否使用简单平均而非元模型
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_simple_average = use_simple_average
        self.base_models_fitted = {}
        self.meta_model_fitted = None

    def fit(self, X, y):
        """训练stacking模型"""
        print("\n=== 开始训练优化Stacking模型 ===")
        print(f"基础模型数量: {len(self.base_models)}")
        print(f"交叉验证折数: {self.n_folds}")
        print(f"策略: {'简单平均' if self.use_simple_average else '元模型学习'}")

        # 准备元特征
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        # 训练基础模型并生成out-of-fold预测
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            print(f"\n训练基础模型: {name}")
            oof_predictions = np.zeros(n_samples)

            # K折交叉验证
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

                # 训练模型
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_fold, y_train_fold)

                # 生成验证集预测
                oof_predictions[val_idx] = model_clone.predict_proba(X_val_fold)[:, 1]

                print(f"  Fold {fold_idx + 1}/{self.n_folds} 完成", end='\r')

            # 保存out-of-fold预测作为元特征
            meta_features[:, model_idx] = oof_predictions

            # 计算该基础模型的性能
            auc = roc_auc_score(y, oof_predictions)
            print(f"  {name} - Out-of-fold AUC: {auc:.4f}")

            # 在全部数据上训练基础模型用于预测
            model_final = self._clone_model(model)
            model_final.fit(X, y)
            self.base_models_fitted[name] = model_final

        # 训练元模型或使用简单平均
        if not self.use_simple_average:
            print("\n训练元模型（带正则化）...")
            self.meta_model_fitted = self._clone_model(self.meta_model)
            self.meta_model_fitted.fit(meta_features, y)
        else:
            print("\n使用简单平均策略...")
            self.meta_model_fitted = None

        print("✓ 优化Stacking模型训练完成")
        return self

    def predict_proba(self, X):
        """预测概率"""
        if not self.base_models_fitted:
            raise ValueError("模型尚未训练，请先调用fit()")

        # 生成基础模型预测作为元特征
        n_samples = X.shape[0]
        n_models = len(self.base_models_fitted)
        meta_features = np.zeros((n_samples, n_models))

        for model_idx, (name, model) in enumerate(self.base_models_fitted.items()):
            meta_features[:, model_idx] = model.predict_proba(X)[:, 1]

        # 使用元模型预测或简单平均
        if not self.use_simple_average and self.meta_model_fitted is not None:
            return self.meta_model_fitted.predict_proba(meta_features)
        else:
            # 简单平均
            avg_proba = np.mean(meta_features, axis=1)
            return np.column_stack([1 - avg_proba, avg_proba])

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)


def train_stacking_v2(X_train, y_train, strategy='ridge', n_folds=5, top_n=3, C=0.1):
    """
    训练优化的stacking v2模型

    参数:
        strategy: 'ridge' (强正则化), 'simple_average' (简单平均), 'xgboost'
        n_folds: 交叉验证折数
        top_n: 使用前N个最强模型
        C: 正则化强度 (越小正则化越强)
    """
    print(f"=== 构建优化Stacking V2模型 ===")
    print(f"策略: {strategy}")
    print(f"Top-N模型: {top_n}")
    print(f"正则化强度C: {C}")

    # 获取基础模型
    base_models = get_top_base_models(top_n=top_n)

    # 获取元模型
    use_simple_average = (strategy == 'simple_average')
    if not use_simple_average:
        meta_model = get_meta_model(model_type=strategy, C=C)
    else:
        meta_model = None

    # 创建并训练stacking模型
    stacking_model = OptimizedStackingClassifier(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=n_folds,
        use_simple_average=use_simple_average
    )

    stacking_model.fit(X_train, y_train)

    # 评估模型
    y_pred = stacking_model.predict(X_train)
    y_pred_proba = stacking_model.predict_proba(X_train)[:, 1]

    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba),
        'train_auprc': average_precision_score(y_train, y_pred_proba),
        'baseline_auprc': y_train.mean(),
        'train_accuracy': accuracy_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_pred),
        'train_f1': f1_score(y_train, y_pred)
    }

    metrics['auprc_improvement'] = (
        (metrics['train_auprc'] / metrics['baseline_auprc'] - 1) * 100
        if metrics['baseline_auprc'] > 0 else 0
    )

    print(f"\n=== 模型性能 ===")
    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"训练集 AUPRC: {metrics['train_auprc']:.4f}")
    print(f"准确率: {metrics['train_accuracy']:.4f}")
    print(f"F1分数: {metrics['train_f1']:.4f}")

    return stacking_model, metrics


def make_stacking_v2_predictions(stacking_model, X_test, sample_path, output_name='stacking_v2'):
    """生成优化stacking模型的预测结果"""
    print("\n=== 生成优化Stacking预测结果 ===")

    # 生成预测
    y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

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

    output_path = f"outputs/{output_name}_submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"✓ 预测结果已保存到: {output_path}")
    print(f"预测统计: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

    return y_pred_proba
