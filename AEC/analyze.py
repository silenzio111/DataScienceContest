from preprocess import *
from model import *
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=VAE().to(device)
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

# model.eval()

features=all_positive.to(device)
with torch.no_grad():
    recon,mu, log_var = model(features)
    # 计算每个正样本的重构损失
    per_sample_pos_loss = get_per_sample_recon_loss(recon, features)
    print(f"正样本数量: {len(per_sample_pos_loss)}")
    print("每个正样本的重构损失:")
    for i, loss in enumerate(per_sample_pos_loss):
        print(f"  正样本 {i}: {loss.item():.4f}")
    print(f"正样本重构损失统计: 均值={per_sample_pos_loss.mean().item():.4f}, 标准差={per_sample_pos_loss.std().item():.4f}")

_negs=[]
for i in range(len(negative_dataset)):
    _negs.append(negative_dataset[i][0])
_negs=torch.stack(_negs).to(device)

with torch.no_grad():
    recon,mu, log_var = model(_negs)
    _, recon_loss, _ = model.loss_function(
        recon, _negs, mu, log_var, beta=0,label=torch.zeros(_negs.shape[0],device=device)
    )
    recon_loss/=_negs.shape[0]
    print(f"负样本总重构误差均值: {recon_loss.mean().item():.4f}")
    
    # 计算每个负样本的重构损失
    per_sample_neg_loss = get_per_sample_recon_loss(recon, _negs)
    print(f"负样本数量: {len(per_sample_neg_loss)}")
    print("每个负样本的重构损失(前20个):")
    for i, loss in enumerate(per_sample_neg_loss[:20]):
        print(f"  负样本 {i}: {loss.item():.4f}")
    print(f"负样本重构损失统计: 均值={per_sample_neg_loss.mean().item():.4f}, 标准差={per_sample_neg_loss.std().item():.4f}")
    
    # 比较正负样本的重构损失分布
    print("\n正负样本重构损失比较:")
    print(f"正样本平均重构损失: {per_sample_pos_loss.mean().item():.4f}")
    print(f"负样本平均重构损失: {per_sample_neg_loss.mean().item():.4f}")
    print(f"正负样本重构损失比值: {per_sample_pos_loss.mean().item() / per_sample_neg_loss.mean().item():.4f}")

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_pos,dataset_neg):
        pos_num=len(dataset_pos)
        self.dataset=[]
        for i in range(pos_num):
            self.dataset.append(dataset_pos[i])
        for i in range(pos_num):
            self.dataset.append(dataset_neg[i])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

val_dataset=ValDataset(positive_dataset,negative_dataset)
val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for batch in val_loader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)  # Move labels to the same device as features
        recon,_, _ = model(features)
        per_sample_loss = get_per_sample_recon_loss(recon, features)
        
        # Assume higher reconstruction loss indicates negative samples
        # First, find a threshold to classify samples
        # For simplicity, we use the median of all reconstruction losses as the threshold
        threshold = torch.median(per_sample_loss)
        
        # Classify samples based on the threshold
        predicted = (per_sample_loss > threshold).int()
        labels = labels.int()  # Ensure labels are integers
        print(f"pos label num: {labels.sum().item()}")
        
        # Calculate accuracy
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        
        print(f"Batch accuracy: {acc:.4f}")



