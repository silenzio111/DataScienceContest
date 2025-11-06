import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CreditDataset(Dataset):
    def __init__(self, csv_file, train=False, scaler=None):
        # 加载CSV文件
        self.data = pd.read_csv(csv_file)
        
        # 确定是否包含target列
        self.has_target = 'target' in self.data.columns
        
        # 分离特征和标签
        if self.has_target:
            self.features = self.data.drop(['id', 'target'], axis=1).values
            self.targets = self.data['target'].values
        else:
            self.features = self.data.drop(['id'], axis=1).values
            self.targets = None
        
        # 数据归一化
        if train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if scaler is None:
                raise ValueError("测试集需要提供训练集的scaler")
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        
        print(f"dim of x: {self.features.shape[1]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.has_target:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features, target
        else:
            return features
    
    def get_positive_stats(self):
        """计算阳性样本数量和占比"""
        if not self.has_target:
            return None, None
        
        positive_count = np.sum(self.targets)
        total_count = len(self.targets)
        positive_ratio = positive_count / total_count
        
        return positive_count, positive_ratio

class PositiveDataset(Dataset):
    """
    仅包含阳性样本的数据集
    """
    def __init__(self, dataset: Dataset):
        """
        初始化阳性数据集
        
        参数:
            dataset: 原始数据集，必须包含target列
        """
        self.dataset = dataset
        self.has_target = dataset.has_target
        
        # 筛选出阳性样本
        if not self.has_target:
            raise ValueError("原始数据集必须包含target列")
        
        # 找到所有阳性样本的索引
        self.positive_indices = np.where(self.dataset.targets == 1)[0]
        
        print(f"阳性样本数量: {len(self.positive_indices)}")
    
    def __len__(self):
        return len(self.positive_indices)
    
    def __getitem__(self, idx):
        # 获取原始数据集的阳性样本
        return self.dataset[self.positive_indices[idx]]

class NegativeDataset(Dataset):
    """
    仅包含阴性样本的数据集
    """
    def __init__(self, dataset: Dataset):
        """
        初始化阴性数据集
        
        参数:
            dataset: 原始数据集，必须包含target列
        """
        self.dataset = dataset
        self.has_target = dataset.has_target
        
        # 筛选出阴性样本
        if not self.has_target:
            raise ValueError("原始数据集必须包含target列")
        
        # 找到所有阴性样本的索引
        self.negative_indices = np.where(self.dataset.targets == 0)[0]
        
        print(f"阴性样本数量: {len(self.negative_indices)}")
    
    def __len__(self):
        return len(self.negative_indices)
    
    def __getitem__(self, idx):
        # 获取原始数据集的阴性样本
        return self.dataset[self.negative_indices[idx]]

class Argumentation(Dataset):
    """
    数据增强类，用于对数据集进行增强处理
    """
    def __init__(self, dataset: Dataset, max_perturbation: float = 0.001, augmentation_factor: int = 5):
        """
        初始化数据增强器
        
        参数:
            dataset: 原始数据集
            max_perturbation: 扰动最大幅度 (0-1之间)，表示对原始数据的最大扰动比例
            augmentation_factor: 增强倍数，表示每个原始样本生成多少个增强样本
        """
        self.dataset = dataset
        self.max_perturbation = max_perturbation
        self.augmentation_factor = augmentation_factor
        self.has_target = dataset.has_target
        
        # 在初始化时就生成所有增强数据
        self.augmented_features = []
        self.augmented_targets = [] if self.has_target else None
        
        # 遍历原始数据集，生成增强样本
        for i in range(len(dataset)):
            # 获取原始样本
            if self.has_target:
                original_sample, target = dataset[i]
            else:
                original_sample = dataset[i]
                target = None
                
            # 为每个原始样本生成多个增强样本
            for _ in range(augmentation_factor):
                # 生成增强样本
                augmented_sample = self.augment_sample(original_sample)
                self.augmented_features.append(augmented_sample)
                
                # 如果有标签，也添加标签
                if self.has_target:
                    self.augmented_targets.append(target)
        
        # 将增强数据转换为tensor
        self.augmented_features = torch.stack(self.augmented_features)
        if self.has_target:
            self.augmented_targets = torch.stack(self.augmented_targets)
        
    def augment_sample(self, sample):
        """
        对单个样本进行增强处理
        
        参数:
            sample: 原始样本 (tensor)
            
        返回:
            增强后的样本
        """
        # 将tensor转换为numpy以便处理
        if isinstance(sample, torch.Tensor):
            sample_np = sample.numpy()
        else:
            sample_np = sample
            
        # 生成与样本形状相同的随机扰动
        perturbation = np.random.uniform(
            -self.max_perturbation, 
            self.max_perturbation, 
            size=sample_np.shape
        )
        
        # 应用扰动
        augmented_sample = sample_np * (1 + perturbation)
        
        # 转换回tensor
        return torch.tensor(augmented_sample, dtype=torch.float32)
    
    def __len__(self):
        """返回增强后的数据集大小"""
        return len(self.augmented_features)
    
    def __getitem__(self, idx):
        """
        获取增强后的样本
        
        参数:
            idx: 索引
            
        返回:
            增强后的样本和标签（如果原始数据集有标签）
        """
        if self.has_target:
            return self.augmented_features[idx], self.augmented_targets[idx]
        else:
            return self.augmented_features[idx]


raw_dataset = CreditDataset('data/train.csv', train=True)
positive_dataset = PositiveDataset(raw_dataset)
negative_dataset = NegativeDataset(raw_dataset)

def get_all_positive():
    l=[]
    for i in range(len(positive_dataset)):
        l.append(positive_dataset[i][0])
    print(f"-阳性样本数量: {len(l)}")
    return torch.stack(l)

all_positive = get_all_positive()

# 主函数，用于演示如何使用
if __name__ == "__main__":
    print("开始处理信用数据集...\n")
    
    # 首先加载训练集来获取scaler
    train_dataset = CreditDataset('data/train.csv', train=True)
    # 加载测试集
    test_dataset = CreditDataset('data/test.csv', train=False, scaler=train_dataset.scaler)
    
    print(f"\n预处理完成，数据已准备好用于模型训练和评估。")
    
    # 演示数据增强功能
    print("\n演示数据增强功能:")
    # 创建数据增强器，扰动幅度为0.1，增强倍数为2
    augmenter = Argumentation(train_dataset, max_perturbation=0.1, augmentation_factor=2)
    
    print(f"原始训练集大小: {len(train_dataset)}")
    print(f"增强后数据集大小: {len(augmenter)}")
    
    # 获取一个原始样本和增强后的样本进行比较
    original_sample, original_target = train_dataset[0]
    augmented_sample, augmented_target = augmenter[0]
    
    print("\n原始样本与增强样本比较:")
    print(f"原始样本前5个特征: {original_sample[:5]}")
    print(f"增强样本前5个特征: {augmented_sample[:5]}")
    print(f"原始标签: {original_target}, 增强标签: {augmented_target}")
    
    # 计算差异
    diff = torch.abs(original_sample - augmented_sample)
    max_diff = torch.max(diff).item()
    avg_diff = torch.mean(diff).item()
    print(f"最大差异: {max_diff:.4f}, 平均差异: {avg_diff:.4f}")
    
    # 创建增强数据集的数据加载器
    augmented_loader = DataLoader(augmenter, batch_size=32, shuffle=True)
    first_augmented_batch = next(iter(augmented_loader))
    if isinstance(first_augmented_batch, list):
        print(f"\n增强数据加载器批次形状: 特征={first_augmented_batch[0].shape}, 标签={first_augmented_batch[1].shape}")
    else:
        print(f"\n增强数据加载器批次形状: {first_augmented_batch.shape}")
    