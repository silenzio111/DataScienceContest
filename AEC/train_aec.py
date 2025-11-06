import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

from preprocess import *
from model import AEC

best_model = None
best_accuracy = 0.0

def train_aec(model, dataloader, test_loader=None, epochs=100, lr=1e-3, alpha=1.0, beta=1.0, device='cpu'):
    """
    训练AEC模型
    
    参数:
        model: AEC模型
        dataloader: 训练数据加载器
        test_loader: 测试数据加载器(可选)
        epochs: 训练轮数
        lr: 学习率
        alpha: 重构损失权重
        beta: 分类损失权重
        device: 设备
    """
    global best_model, best_accuracy
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    recon_losses = []
    cls_losses = []
    accuracies = []
    test_recon_losses = []  # 记录测试集重构损失
    
    print("开始训练AEC模型...")
    model.train()
    
    for epoch in range(epochs):
        total_loss_epoch = 0
        recon_loss_epoch = 0
        cls_loss_epoch = 0
        correct_predictions = 0
        total_samples = 0
        
        # 训练阶段1: 在训练数据上训练编码器、解码器和分类器
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device).long()
            
            # 前向传播
            optimizer.zero_grad()
            recon_batch, logits, z = model(data)
            
            # 计算损失
            total_loss, recon_loss, cls_loss = model.loss_function(
                recon_batch, data, logits, labels, alpha=alpha, beta=beta
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
            cls_loss_epoch += cls_loss.item()
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # 训练阶段2: 使用测试数据训练编码器和解码器（无监督学习）
        if test_loader is not None:
            for test_batch in test_loader:
                # 处理测试数据，可能没有标签
                if isinstance(test_batch, (list, tuple)) and len(test_batch) == 2:
                    test_data, _ = test_batch  # 有标签的情况
                else:
                    test_data = test_batch  # 无标签的情况
                
                test_data = test_data.to(device)
                
                # 前向传播（只使用重构部分）
                optimizer.zero_grad()
                test_recon_batch, _, _ = model(test_data)
                
                # 只计算重构损失（不使用分类损失）
                test_recon_loss = F.mse_loss(test_recon_batch, test_data)
                
                # Add L1 regularization term
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, p=1)
                test_total_loss = test_recon_loss + l1_lambda * l1_reg
                
                # 反向传播
                test_total_loss.backward()
                optimizer.step()
        
        # 计算平均损失和准确率
        avg_total_loss = total_loss_epoch / len(dataloader.dataset)
        avg_recon_loss = recon_loss_epoch / len(dataloader.dataset)
        avg_cls_loss = cls_loss_epoch / len(dataloader.dataset)
        accuracy = correct_predictions / total_samples
        
        train_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        cls_losses.append(avg_cls_loss)
        accuracies.append(accuracy)
        
        # 计算测试集重构损失（用于监控）
        test_recon_loss_epoch = 0
        if test_loader is not None:
            model.eval()  # 切换到评估模式
            with torch.no_grad():
                for test_batch in test_loader:
                    # 处理测试数据，可能没有标签
                    if isinstance(test_batch, (list, tuple)) and len(test_batch) == 2:
                        test_data, _ = test_batch  # 有标签的情况
                    else:
                        test_data = test_batch  # 无标签的情况
                    
                    test_data = test_data.to(device)
                    test_recon_batch, _, _ = model(test_data)
                    # 计算重构损失 (MSE)
                    test_recon_loss = F.mse_loss(test_recon_batch, test_data, reduction='sum')
                    test_recon_loss_epoch += test_recon_loss.item()
            
            # 计算平均测试重构损失
            avg_test_recon_loss = test_recon_loss_epoch / len(test_loader.dataset)
            test_recon_losses.append(avg_test_recon_loss)
            model.train()  # 切换回训练模式
        else:
            test_recon_losses.append(0)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict().copy()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            test_loss_str = f', Test Recon Loss: {avg_test_recon_loss:.4f}' if test_loader is not None else ''
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Total Loss: {avg_total_loss:.4f}, '
                  f'Recon Loss: {avg_recon_loss:.4f}, '
                  f'Cls Loss: {avg_cls_loss:.4f}, '
                  f'Accuracy: {accuracy:.4f}'
                  f'{test_loss_str}')
    
    print("训练完成!")
    return train_losses, recon_losses, cls_losses, accuracies, test_recon_losses

def evaluate_model(model, dataloader, device='cpu'):
    """
    评估模型性能
    
    参数:
        model: 训练好的AEC模型
        dataloader: 数据加载器
        device: 设备
    """
    model.eval()
    
    total_loss = 0
    recon_loss = 0
    cls_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    all_embeddings = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device).long()
            
            # 前向传播
            recon_batch, logits, z = model(data)
            
            # 计算损失
            total_loss_batch, recon_loss_batch, cls_loss_batch = model.loss_function(
                recon_batch, data, logits, labels
            )
            
            # 累计损失
            total_loss += total_loss_batch.item()
            recon_loss += recon_loss_batch.item()
            cls_loss += cls_loss_batch.item()
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # 保存嵌入和标签用于可视化
            all_embeddings.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # 计算平均值
    avg_total_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = recon_loss / len(dataloader.dataset)
    avg_cls_loss = cls_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_samples
    
    # 合并所有批次
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    print(f"评估结果:")
    print(f"总损失: {avg_total_loss:.4f}")
    print(f"重构损失: {avg_recon_loss:.4f}")
    print(f"分类损失: {avg_cls_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    
    return embeddings, labels, preds, accuracy

def visualize_latent_space(embeddings, labels, preds, save_path='metrics/aec_latent_space.png'):
    """
    可视化潜在空间
    
    参数:
        embeddings: 潜在表示
        labels: 真实标签
        preds: 预测标签
        save_path: 保存路径
    """
    # 使用t-SNE降维到2D进行可视化
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(1, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 真实标签可视化
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], 
        c=labels, cmap='viridis', alpha=0.7
    )
    plt.colorbar(scatter1, label='True Label')
    plt.title('True Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 预测标签可视化
    plt.subplot(1, 3, 2)
    scatter2 = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], 
        c=preds, cmap='viridis', alpha=0.7
    )
    plt.colorbar(scatter2, label='Predicted Label')
    plt.title('Predicted Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 错误分类可视化
    plt.subplot(1, 3, 3)
    errors = (labels != preds).astype(int)
    scatter3 = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], 
        c=errors, cmap='coolwarm', alpha=0.7
    )
    plt.colorbar(scatter3, label='Classification Error')
    plt.title('Classification Errors')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"潜在空间可视化已保存到 {save_path}")

def plot_training_curves(train_losses, recon_losses, cls_losses, accuracies, test_recon_losses=None, save_path='metrics/aec_training_curves.png'):
    """
    绘制训练曲线
    
    参数:
        train_losses: 总损失列表
        recon_losses: 重构损失列表
        cls_losses: 分类损失列表
        accuracies: 准确率列表
        test_recon_losses: 测试集重构损失列表(可选)
        save_path: 保存路径
    """
    # 根据是否有测试集重构损失调整子图布局
    if test_recon_losses is not None:
        plt.figure(figsize=(15, 10))
    else:
        plt.figure(figsize=(12, 8))
    
    # 总损失
    if test_recon_losses is not None:
        plt.subplot(3, 2, 1)
    else:
        plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 重构损失
    if test_recon_losses is not None:
        plt.subplot(3, 2, 2)
    else:
        plt.subplot(2, 2, 2)
    plt.plot(recon_losses, label='Train')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 分类损失
    if test_recon_losses is not None:
        plt.subplot(3, 2, 3)
    else:
        plt.subplot(2, 2, 3)
    plt.plot(cls_losses)
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 准确率
    if test_recon_losses is not None:
        plt.subplot(3, 2, 4)
    else:
        plt.subplot(2, 2, 4)
    plt.plot(accuracies)
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # 如果有测试集重构损失，添加额外的子图
    if test_recon_losses is not None:
        # 绘制训练集和测试集重构损失对比
        plt.subplot(3, 2, 5)
        plt.plot(recon_losses, label='Train Reconstruction Loss')
        plt.plot(test_recon_losses, label='Test Reconstruction Loss')
        plt.title('Train vs Test Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # 绘制训练和测试重构损失差值
        plt.subplot(3, 2, 6)
        diff = np.array(test_recon_losses) - np.array(recon_losses)
        plt.plot(diff)
        plt.title('Test - Train Reconstruction Loss Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到 {save_path}")

def main():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_dataset = raw_dataset
    print(f"训练集样本数量: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 加载测试数据（使用训练集的scaler）
    test_dataset = CreditDataset('data/test.csv', train=False, scaler=train_dataset.scaler)
    print(f"测试集样本数量: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建AEC模型
    input_dim = train_dataset[0][0].shape[0]
    latent_dim = 5
    model = AEC(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    # 打印模型信息
    print(f"输入维度: {input_dim}")
    print(f"潜在维度: {latent_dim}")
    print(f"模型参数量: {model.count_parameters()}")
    
    # 训练模型
    train_losses, recon_losses, cls_losses, accuracies, test_recon_losses = train_aec(
        model, train_loader, test_loader, epochs=200, lr=0.001, alpha=1.0, beta=1.0, device=device
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, recon_losses, cls_losses, accuracies, test_recon_losses)
    
    # 评估模型
    embeddings, labels, preds, accuracy = evaluate_model(model, train_loader, device=device)
    
    # 可视化潜在空间
    visualize_latent_space(embeddings, labels, preds)
    
    # 保存模型
    os.makedirs('model', exist_ok=True)
    torch.save(best_model, f'model/aec_model_{best_accuracy:.4f}.pth')
    print(f"最佳模型已保存到 model/aec_model_{best_accuracy:.4f}.pth")
    
    # 保存当前模型
    torch.save(model.state_dict(), 'model/aec_model.pth')
    print("模型已保存到 model/aec_model.pth")
    
    # 测试降维和分类功能
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        sample_data, sample_labels = next(iter(train_loader))
        sample_data = sample_data.to(device)
        
        # 降维和分类
        embeddings = model.get_embedding(sample_data)
        _, logits, _ = model(sample_data)
        preds = torch.argmax(logits, dim=1)
        
        print(f"\n测试结果:")
        print(f"原始数据形状: {sample_data.shape}")
        print(f"降维后形状: {embeddings.shape}")
        print(f"真实标签: {sample_labels[:5].tolist()}")
        print(f"预测标签: {preds[:5].tolist()}")
        print(f"降维后前5个样本的前5个维度:")
        print(embeddings[:5, :5])

if __name__ == "__main__":
    main()