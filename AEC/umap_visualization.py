import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.manifold import TSNE

from preprocess import CreditDataset, PositiveDataset, NegativeDataset
from model import VAE

def load_model_and_data(model_path='model/vae_model.pth'):
    """
    加载训练好的VAE模型和数据
    
    参数:
        model_path: 模型路径
        
    返回:
        model: 加载的VAE模型
        dataset: 原始数据集
        positive_dataset: 正样本数据集
        negative_dataset: 负样本数据集
    """
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    dataset = CreditDataset('data/train.csv', train=True)
    positive_dataset = PositiveDataset(dataset)
    negative_dataset = NegativeDataset(dataset)
    
    # 创建模型
    input_dim = dataset[0][0].shape[0]
    latent_dim = 5  # 与训练时保持一致
    model = VAE()
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"数据加载完成:")
    print(f"-总样本数: {len(dataset)}")
    print(f"-正样本数: {len(positive_dataset)}")
    print(f"-负样本数: {len(negative_dataset)}")
    
    return model, dataset, positive_dataset, negative_dataset, device

def get_diff(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            # 获取潜在表示
            recon,_,_ = model(data)
            diff=(recon-data)**2
            all_embeddings.append(diff.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # 合并所有批次
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels

def apply_tsne(embeddings, perplexity=30, n_components=2):
    """
    使用t-SNE进行降维
    
    参数:
        embeddings: 高维嵌入
        perplexity: t-SNE的perplexity参数
        n_components: 降维后的维度
        
    返回:
        embeddings_2d: 降维后的嵌入
    """
    print(f"应用t-SNE降维，参数: perplexity={perplexity}")
    
    # 确保perplexity小于样本数量
    n_samples = embeddings.shape[0]
    perplexity = min(perplexity, max(1, n_samples // 3))
    
    tsne = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    return embeddings_2d

def visualize_tsne_results(embeddings_2d, labels, save_path='metrics/tsne_visualization.png'):  
    """
    可视化t-SNE降维结果
    
    参数:
        embeddings_2d: 2D嵌入
        labels: 标签
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 创建DataFrame便于绘图
    df = pd.DataFrame({
        'tSNE_1': embeddings_2d[:, 0],
        'tSNE_2': embeddings_2d[:, 1],
        'Label': labels
    })
    
    # 将标签转换为字符串，便于绘图
    df['Label_Str'] = df['Label'].map({0: 'Negative Samples', 1: 'Positive Samples'})
    
    # 使用seaborn绘制散点图
    sns.scatterplot(
        data=df,
        x='tSNE_1',
        y='tSNE_2',
        hue='Label_Str',
        palette=['blue', 'red'],
        alpha=0.7,
        s=50
    )
    
    plt.title('t-SNE Dimensionality Reduction Visualization: Positive and Negative Samples Distribution', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(title='Sample Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"t-SNE可视化结果已保存到 {save_path}")
    
    # 打印一些统计信息
    print("\n样本分布统计:")
    print(f"-正样本数量: {np.sum(labels)}")
    print(f"-负样本数量: {len(labels) - np.sum(labels)}")

def apply_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    使用UMAP进行降维
    
    参数:
        embeddings: 高维嵌入
        n_neighbors: UMAP的n_neighbors参数
        min_dist: UMAP的min_dist参数
        n_components: 降维后的维度
        
    返回:
        embeddings_2d: 降维后的嵌入
    """
    print(f"应用UMAP降维，参数: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    return embeddings_2d

def visualize_umap_results(embeddings_2d, labels, save_path='metrics/umap_visualization.png'):  
    """
    可视化UMAP降维结果
    
    参数:
        embeddings_2d: 2D嵌入
        labels: 标签
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 创建DataFrame便于绘图
    df = pd.DataFrame({
        'UMAP_1': embeddings_2d[:, 0],
        'UMAP_2': embeddings_2d[:, 1],
        'Label': labels
    })
    
    # 将标签转换为字符串，便于绘图
    df['Label_Str'] = df['Label'].map({0: 'Negative Samples', 1: 'Positive Samples'})
    
    # 使用seaborn绘制散点图
    sns.scatterplot(
        data=df,
        x='UMAP_1',
        y='UMAP_2',
        hue='Label_Str',
        palette=['blue', 'red'],
        alpha=0.7,
        s=50
    )
    
    plt.title('UMAP Dimensionality Reduction Visualization: Positive and Negative Samples Distribution', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.legend(title='Sample Type', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"UMAP可视化结果已保存到 {save_path}")
    
    # 打印一些统计信息
    print("\n样本分布统计:")
    print(f"-正样本数量: {np.sum(labels)}")
    print(f"-负样本数量: {len(labels) - np.sum(labels)}")

def main():
    # 加载模型和数据
    model, dataset, positive_dataset, negative_dataset, device = load_model_and_data()
    
    # 获取潜在表示
    print("\n获取diff...")
    embeddings, labels = get_diff(model, dataset, device)
    print(f"潜在表示形状: {embeddings.shape}")
    
    # 应用UMAP降维
    print("\n应用UMAP降维...")
    embeddings_2d = apply_umap(embeddings, n_neighbors=15, min_dist=0.1)
    print(f"UMAP降维后形状: {embeddings_2d.shape}")
    
    # 可视化结果
    print("\n可视化UMAP降维结果...")
    visualize_umap_results(embeddings_2d, labels)
    
    # 应用t-SNE降维
    print("\n应用t-SNE降维...")
    embeddings_2d_tsne = apply_tsne(embeddings, perplexity=min(30, max(1, len(embeddings) // 3)), n_components=2)
    
    # 可视化t-SNE结果
    print("\n可视化t-SNE降维结果...")
    visualize_tsne_results(embeddings_2d_tsne, labels, 'metrics/tsne_visualization.png')
    
    # 尝试不同的UMAP参数
    print("\n尝试不同的UMAP参数...")
    
    # 参数组合1: 更大的n_neighbors
    embeddings_2d_1 = apply_umap(embeddings, n_neighbors=30, min_dist=0.1)
    visualize_umap_results(embeddings_2d_1, labels, save_path='metrics/umap_visualization_n30.png')
    
    # 参数组合2: 更小的min_dist
    embeddings_2d_2 = apply_umap(embeddings, n_neighbors=15, min_dist=0.05)
    visualize_umap_results(embeddings_2d_2, labels, save_path='metrics/umap_visualization_md005.png')
    
    # 参数组合3: 3D可视化
    embeddings_3d = apply_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=3)
    
    # 3D可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 分别绘制正样本和负样本
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    ax.scatter(
        embeddings_3d[negative_mask, 0],
        embeddings_3d[negative_mask, 1],
        embeddings_3d[negative_mask, 2],
        c='blue', label='Negative Samples', alpha=0.7, s=30
    )
    
    ax.scatter(
        embeddings_3d[positive_mask, 0],
        embeddings_3d[positive_mask, 1],
        embeddings_3d[positive_mask, 2],
        c='red', label='Positive Samples', alpha=0.7, s=30
    )
    
    ax.set_title('UMAP 3D Dimensionality Reduction Visualization: Positive and Negative Samples Distribution', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_zlabel('UMAP Dimension 3', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('metrics/umap_3d_visualization.png', dpi=300)
    plt.close()
    
    print("3D UMAP可视化结果已保存到 umap_3d_visualization.png")
    print("\n所有可视化完成!")

if __name__ == "__main__":
    main()