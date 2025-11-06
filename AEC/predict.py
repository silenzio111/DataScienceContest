import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from preprocess import *
from model import VAE

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
model = VAE().to(device)
model.load_state_dict(torch.load('model/vae_model_1.0810.pth'))
model.eval()

def get_per_sample_recon_loss(recon_x, x):
    """
    计算每个样本的重构损失
    
    参数:
        recon_x: 重构数据
        x: 原始数据
        
    返回:
        每个样本的重构损失
    """
    # 使用reduction='none'计算每个元素的MSE
    elementwise_loss = F.mse_loss(recon_x, x, reduction='none')
    # 沿特征维度求和，得到每个样本的总损失
    per_sample_loss = torch.sum(elementwise_loss, dim=1)
    return per_sample_loss

# 加载测试数据
test_data = pd.read_csv('data/test.csv')
print(f"测试数据形状: {test_data.shape}")

# 提取特征列（排除id列）
test_features = test_data.drop(columns=['id']).values
test_features = torch.tensor(test_features, dtype=torch.float32).to(device)

# 计算测试集的重构损失
with torch.no_grad():
    recon, mu, log_var = model(test_features)
    test_recon_loss = get_per_sample_recon_loss(recon, test_features)
    
    # 获取训练数据的重构损失作为参考
    # 正样本
    pos_features = all_positive.to(device)
    pos_recon, pos_mu, pos_log_var = model(pos_features)
    pos_recon_loss = get_per_sample_recon_loss(pos_recon, pos_features)
    
    # 负样本
    neg_features = []
    for i in range(len(negative_dataset)):
        neg_features.append(negative_dataset[i][0])
    neg_features = torch.stack(neg_features).to(device)
    neg_recon, neg_mu, neg_log_var = model(neg_features)
    neg_recon_loss = get_per_sample_recon_loss(neg_recon, neg_features)
    
    # 计算阈值 - 使用正负样本重构损失的均值作为阈值
    pos_mean = pos_recon_loss.mean().item()
    neg_mean = neg_recon_loss.mean().item()
    threshold = (pos_mean + neg_mean) / 2
    
    print(f"正样本平均重构损失: {pos_mean:.4f}")
    print(f"负样本平均重构损失: {neg_mean:.4f}")
    print(f"分类阈值: {threshold:.4f}")
    
    # 根据阈值预测测试集的目标值
    # 重构损失大于阈值认为是负样本(0)，小于阈值认为是正样本(1)
    predictions = (test_recon_loss < threshold).int().cpu().numpy()
    
    # 统计预测结果
    pos_count = np.sum(predictions == 1)
    neg_count = np.sum(predictions == 0)
    print(f"预测正样本数量: {pos_count}")
    print(f"预测负样本数量: {neg_count}")
    print(f"正样本比例: {pos_count / len(predictions) * 100:.2f}%")

# 创建提交文件
submission = pd.DataFrame({
    'id': test_data['id'],
    'target': predictions
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("预测结果已保存到 submission.csv")

# 显示前10个预测结果
print("\n前10个预测结果:")
print(submission.head(10))