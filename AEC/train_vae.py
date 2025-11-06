import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from preprocess import *
from model import VAE

best_model=None
last_diff=-1919810

def train_vae(model, dataloader, epochs=100, lr=1e-3, beta=1.0, device='cpu'):
    global best_model, last_diff
    """
    训练VAE模型
    
    参数:
        model: VAE模型
        dataloader: 数据加载器
        epochs: 训练轮数
        lr: 学习率
        beta: KL散度权重
        device: 设备
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    recon_losses = []
    kl_losses = []
    positive_losses = []
    
    print("开始训练VAE模型...")
    model.train()
    
    for epoch in range(epochs):
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            
            # 计算损失
            total_loss, recon_loss, kl_loss = model.loss_function(
                recon_batch, data, mu, log_var, beta=beta,label=label
            )

            # Add L1 regularization term
            l1_lambda = 0.01  # Regularization strength
            l1_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l1_reg += torch.norm(param, p=1)
            total_loss += l1_lambda * l1_reg
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录损失
            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()
        
        # 计算平均损失
        avg_total_loss = total_loss_epoch / len(dataloader.dataset)
        avg_recon_loss = recon_loss_epoch / len(dataloader.dataset)
        avg_kl_loss = kl_loss_epoch / len(dataloader.dataset)
        
        train_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        # -------------------惩罚正样本 - 只惩罚loss最小的前一半样本
        data=all_positive.to(device)
        recon_batch, mu, log_var = model(data)
        optimizer.zero_grad()
        
        # 计算每个正样本的重构损失
        elementwise_loss = F.mse_loss(recon_batch, data, reduction='none')
        per_sample_loss = torch.sum(elementwise_loss, dim=1)  # 每个样本的总损失
        
        # 找出损失最小的前一半样本的索引
        num_samples = len(data)
        half_samples = num_samples 
        _, min_loss_indices = torch.topk(per_sample_loss, half_samples, largest=False)
        
        # 创建掩码，只对损失最小的前一半样本进行惩罚
        mask = torch.zeros(num_samples, device=device)
        mask[min_loss_indices] = 1.0
        
        # 计算加权损失，只对损失最小的前一半样本进行反向传播
        weighted_loss = per_sample_loss * mask
        loss_pos = -torch.sum(weighted_loss) * 10  # 负号表示要增加重构误差
        # loss_pos.backward()
        # optimizer.step()
        #----------------------------
        
        # 计算实际被惩罚样本的平均损失
        pos_recon_loss = torch.sum(weighted_loss).item() / half_samples
        positive_losses.append(pos_recon_loss)
        
        diff=pos_recon_loss-avg_recon_loss
        if diff>last_diff:
            best_model=model.state_dict().copy()
            last_diff=diff

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Total Loss: {avg_total_loss:.4f}, '
                  f'Recon Loss: {avg_recon_loss:.4f}, '
                  f'KL Loss: {avg_kl_loss:.4f}, '
                  f'Positive Loss: {pos_recon_loss:.4f}')
    
    print("训练完成!")
    return train_losses, recon_losses, kl_losses, positive_losses

def visualize_latent_space(model, dataloader, device='cpu', save_path='metrics/latent_space.png'):
    """
    可视化潜在空间
    
    参数:
        model: 训练好的VAE模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            # 获取潜在表示
            embeddings = model.get_embedding(data)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # 合并所有批次
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # 使用t-SNE降维到2D进行可视化
    # 确保perplexity小于样本数量，通常设置为样本数量的1/3到1/5之间
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(1, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], 
        c=labels, cmap='viridis', alpha=0.7
    )
    plt.colorbar(scatter, label='Label')
    plt.title('VAE Latent Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(save_path)
    plt.close()
    
    print(f"潜在空间可视化已保存到 {save_path}")

def visualize_latent_space_with_positive(model, negative_loader, positive_dataset, device='cpu', save_path='metrics/latent_space_with_positive.png'):
    """
    可视化潜在空间，同时显示负样本和正样本
    
    参数:
        model: 训练好的VAE模型
        negative_loader: 负样本数据加载器
        positive_dataset: 正样本数据集
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    
    # 获取负样本的潜空间表示
    negative_embeddings = []
    with torch.no_grad():
        for data, _ in negative_loader:
            data = data.to(device)
            embeddings = model.get_embedding(data)
            negative_embeddings.append(embeddings.cpu().numpy())
    
    negative_embeddings = np.concatenate(negative_embeddings, axis=0)
    
    # 获取正样本的潜空间表示
    positive_embeddings = []
    with torch.no_grad():
        for i in range(len(positive_dataset)):
            data, _ = positive_dataset[i]
            data = data.unsqueeze(0).to(device)  # 添加batch维度
            embedding = model.get_embedding(data)
            positive_embeddings.append(embedding.cpu().numpy())
    
    positive_embeddings = np.concatenate(positive_embeddings, axis=0)
    
    # 合并所有样本
    all_embeddings = np.vstack([negative_embeddings, positive_embeddings])
    labels = np.hstack([np.zeros(len(negative_embeddings)), np.ones(len(positive_embeddings))])
    
    # 使用t-SNE降维到2D进行可视化
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, max(1, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # 分离正负样本的2D表示
    negative_2d = embeddings_2d[:len(negative_embeddings)]
    positive_2d = embeddings_2d[len(negative_embeddings):]
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制负样本
    plt.scatter(
        negative_2d[:, 0], negative_2d[:, 1], 
        c='blue', alpha=0.6, label='Negative Samples', s=30
    )
    
    # 绘制正样本
    plt.scatter(
        positive_2d[:, 0], positive_2d[:, 1], 
        c='red', alpha=0.8, label='Positive Samples', s=50
    )
    
    plt.title('VAE Latent Space Visualization: Positive vs Negative Samples', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(
        0.02, 0.98, 
        f'Negative Samples: {len(negative_embeddings)}\nPositive Samples: {len(positive_embeddings)}', 
        transform=plt.gca().transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"正负样本潜空间可视化已保存到 {save_path}")
    print(f"负样本数量: {len(negative_embeddings)}")
    print(f"正样本数量: {len(positive_embeddings)}")

def plot_training_curves(train_losses, recon_losses, kl_losses, positive_losses, save_path='metrics/training_curves.png'):
    """
    绘制训练曲线
    
    参数:
        train_losses: 总损失列表
        recon_losses: 重构损失列表
        kl_losses: KL散度损失列表
        positive_losses: 阳性损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 10))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 重构损失
    plt.subplot(2, 2, 3)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # KL散度损失
    plt.subplot(2, 2, 2)
    plt.plot(kl_losses)
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 阳性损失
    plt.subplot(2, 2, 4)
    plt.plot(positive_losses)
    plt.title('Positive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"训练曲线已保存到 {save_path}")

def main():
    import os
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_dataset = raw_dataset
    print(f"训练集样本数量: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建VAE模型
    input_dim = train_dataset[0][0].shape[0]
    latent_dim = 5
    model = VAE().to(device)
    if os.path.exists('model/vae_model.pth'):
        model.load_state_dict(torch.load('model/vae_model.pth'))
        print("已加载预训练模型")
        #test pos avg  loss
        features=all_positive
        features = features.to(device)
        with torch.no_grad():
            recon,mu, log_var = model(features)
            # 计算重构误差
            total_loss, recon_loss, kl_loss = model.loss_function(
                recon, features, mu, log_var, beta=0,label=torch.ones(features.shape[0],device=device)
            )
            recon_loss/=features.shape[0]
            print(f"pos样本重构误差均值: {recon_loss.mean().item():.4f}")
    
    # 打印模型信息
    print(f"输入维度: {input_dim}")
    print(f"潜在维度: {latent_dim}")
    print(f"模型参数量: {model.count_parameters()}")
    
    # 训练模型
    train_losses, recon_losses, kl_losses, positive_losses = train_vae(
        model, train_loader, epochs=400, lr=0.0005, beta=2, device=device
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, recon_losses, kl_losses, positive_losses)
    
    # 可视化潜在空间
    visualize_latent_space(model, train_loader, device=device)
    
    # 可视化正负样本的潜空间表示
    visualize_latent_space_with_positive(model, train_loader, positive_dataset, device=device)
    
    # 保存模型
    os.makedirs('model', exist_ok=True)
    torch.save(best_model, f'model/vae_model_{last_diff:.4f}.pth')
    print("模型已保存到 model/vae_model.pth")

    # save raw
    torch.save(model.state_dict(), f'model/vae_model.pth')
    print("模型已保存到 model/vae_model.pth")
    
    # 测试降维功能
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        sample_data, sample_labels = next(iter(train_loader))
        sample_data = sample_data.to(device)
        
        # 降维
        embeddings = model.get_embedding(sample_data)
        
        print(f"\n降维测试:")
        print(f"原始数据形状: {sample_data.shape}")
        print(f"降维后形状: {embeddings.shape}")
        print(f"样本标签: {sample_labels[:5].tolist()}")
        print(f"降维后前5个样本的前5个维度:")
        print(embeddings[:5, :5])

if __name__ == "__main__":
    main()
    import umap_visualization
    umap_visualization.main()