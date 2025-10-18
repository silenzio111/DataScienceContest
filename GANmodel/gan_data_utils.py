"""
GANmodel独立数据处理工具
从baseline提取必要功能，保持GANmodel独立性
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """加载单个数据文件"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_excel(file_path)
    print(f"数据形状: {df.shape}")
    if 'target' in df.columns:
        print(f"违约率: {df['target'].mean():.4f}")
    return df


def identify_features(df):
    """识别特征类型"""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除不应作为特征的字段
    exclude_features = ['id', 'target']
    numeric_features = [col for col in numeric_features if col not in exclude_features]

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_features:
        categorical_features.remove('id')

    return numeric_features, categorical_features


def handle_missing_values(df):
    """处理缺失值 - 简化版接口"""
    df_processed = df.copy()

    # 识别特征类型
    numeric_features, categorical_features = identify_features(df)

    # 处理数值特征
    if numeric_features:
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_features] = imputer.fit_transform(df_processed[numeric_features])

    # 处理类别特征
    if categorical_features:
        imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = imputer.fit_transform(df_processed[categorical_features])

    return df_processed


def create_features(train_df, test_df):
    """特征工程 - 为训练集和测试集创建相同的特征"""
    train_fe = train_df.copy()
    test_fe = test_df.copy()

    # 识别数值特征
    numeric_features, _ = identify_features(train_df)

    for df in [train_fe, test_fe]:
        # 负债收入比
        if 'amount' in df.columns and 'income' in df.columns:
            df['debt_to_income_ratio'] = df['amount'] / (df['income'] + 1e-6)

        # 违约次数平方
        if 'total_default_number' in df.columns:
            df['default_squared'] = df['total_default_number'] ** 2

    # 分离特征和标签
    if 'target' in train_fe.columns:
        y_train = train_fe['target']
        X_train = train_fe.drop('target', axis=1)
        if 'id' in X_train.columns:
            X_train = X_train.drop('id', axis=1)
    else:
        raise ValueError("训练数据缺少target列")

    X_test = test_fe.copy()
    if 'id' in X_test.columns:
        X_test = X_test.drop('id', axis=1)

    # 确保测试集和训练集特征一致
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0

    extra_cols = set(X_test.columns) - set(X_train.columns)
    for col in extra_cols:
        X_test = X_test.drop(col, axis=1)

    # 确保列顺序一致
    X_test = X_test[X_train.columns]

    return X_train, X_test, y_train


def balance_samples(X_train, y_train, method='smote'):
    """样本平衡 - 使用SMOTE"""
    if method == 'smote':
        print(f"处理前类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"处理后类别分布: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        return X_resampled, y_resampled
    return X_train, y_train
