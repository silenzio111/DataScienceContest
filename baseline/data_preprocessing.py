"""
数据预处理函数式模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    """加载数据"""
    print("正在加载数据...")
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    return train_df, test_df

def explore_data(df, data_name="数据集"):
    """探索数据"""
    print(f"\n=== {data_name} 基本信息 ===")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    if 'target' in df.columns:
        print(f"\n目标变量分布:")
        print(df['target'].value_counts())
        print(f"违约率: {df['target'].mean():.4f}")

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

def handle_missing_values(df, numeric_features, categorical_features):
    """处理缺失值"""
    df_processed = df.copy()
    if numeric_features:
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_features] = imputer.fit_transform(df_processed[numeric_features])
    if categorical_features:
        imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = imputer.fit_transform(df_processed[categorical_features])
    return df_processed

def encode_categorical(df, categorical_features, encoders=None, is_train=True):
    """编码类别型特征"""
    df_encoded = df.copy()
    if encoders is None:
        encoders = {}

    for feature in categorical_features:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            if is_train:
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                encoders[feature] = le
            else:
                if feature in encoders:
                    le = encoders[feature]
                    # 处理新类别
                    unique_values = set(df_encoded[feature].astype(str))
                    train_values = set(le.classes_)
                    new_values = unique_values - train_values
                    if new_values:
                        df_encoded[feature] = df_encoded[feature].astype(str)
                        df_encoded.loc[~df_encoded[feature].isin(le.classes_), feature] = 'Unknown'
                        all_values = list(le.classes_) + ['Unknown']
                        le.fit(all_values)
                    df_encoded[feature] = le.transform(df_encoded[feature].astype(str))

    return df_encoded, encoders

def create_features(df, numeric_features):
    """创建新特征"""
    df_fe = df.copy()
    available_features = [col for col in numeric_features if col in df_fe.columns]

    if len(available_features) > 0:
        # 负债收入比
        if 'amount' in available_features and 'income' in available_features:
            df_fe['debt_to_income_ratio'] = df_fe['amount'] / (df_fe['income'] + 1e-6)

        # 违约次数平方
        if 'total_default_number' in available_features:
            df_fe['default_squared'] = df_fe['total_default_number'] ** 2

    return df_fe

def scale_features(X_train, X_test, numeric_features):
    """特征缩放"""
    scaler = StandardScaler()
    available_features = [col for col in numeric_features if col in X_train.columns]

    if available_features:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[available_features] = scaler.fit_transform(X_train_scaled[available_features])
        X_test_scaled[available_features] = scaler.transform(X_test_scaled[available_features])
        return X_train_scaled, X_test_scaled, scaler

    return X_train, X_test, None

def balance_samples(X_train, y_train, method='smote'):
    """样本平衡"""
    if method == 'smote':
        print(f"处理前类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"处理后类别分布: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        return X_resampled, y_resampled
    return X_train, y_train

def preprocess_pipeline(train_path, test_path):
    """完整的预处理流水线"""
    print("=== 开始数据预处理 ===")

    # 加载数据
    train_df, test_df = load_data(train_path, test_path)
    explore_data(train_df, "训练集")
    explore_data(test_df, "测试集")

    # 识别特征
    numeric_features, categorical_features = identify_features(train_df)
    print(f"数值特征: {len(numeric_features)}, 类别特征: {len(categorical_features)}")

    # 处理缺失值
    train_clean = handle_missing_values(train_df, numeric_features, categorical_features)
    test_clean = handle_missing_values(test_df, numeric_features, categorical_features)

    # 特征工程
    train_fe = create_features(train_clean, numeric_features)
    test_fe = create_features(test_clean, numeric_features)

    # 更新数值特征列表
    new_numeric_features, categorical_features = identify_features(train_fe)

    # 编码类别特征
    train_encoded, encoders = encode_categorical(train_fe, categorical_features, is_train=True)
    test_encoded, _ = encode_categorical(test_fe, categorical_features, encoders, is_train=False)

    # 分离特征和标签
    if 'target' in train_encoded.columns:
        y_train = train_encoded['target']
        X_train = train_encoded.drop('target', axis=1)
        # 明确排除 id 特征
        if 'id' in X_train.columns:
            X_train = X_train.drop('id', axis=1)
    else:
        return None

    X_test = test_encoded
    # 明确排除 id 特征
    if 'id' in X_test.columns:
        X_test = X_test.drop('id', axis=1)

    # 特征缩放
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, new_numeric_features)

    # 样本平衡
    X_balanced, y_balanced = balance_samples(X_train_scaled, y_train)

    print("=== 预处理完成 ===")
    print(f"最终训练集形状: {X_balanced.shape}")
    print(f"最终测试集形状: {X_test_scaled.shape}")

    return X_balanced, X_test_scaled, y_balanced