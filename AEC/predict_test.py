import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from preprocess import CreditDataset
from model import AEC

def predict_test_data(model_path, test_csv_path, output_csv_path, device='cpu'):
    """
    使用训练好的AEC模型对测试数据进行预测
    
    参数:
        model_path: 训练好的模型路径
        test_csv_path: 测试数据CSV文件路径
        output_csv_path: 输出预测结果CSV文件路径
        device: 计算设备
    """
    # 加载训练数据集以获取scaler
    print("加载训练数据集以获取scaler...")
    train_dataset = CreditDataset('data/train.csv', train=True)
    
    # 加载测试数据集
    print("加载测试数据集...")
    test_dataset = CreditDataset(test_csv_path, train=False, scaler=train_dataset.scaler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = AEC(input_dim=22, latent_dim=5, hidden_dims=[14, 8, 6], num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 收集所有测试数据的ID和负样本似然值
    test_ids = []
    negative_likelihoods = []
    
    print("开始预测...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # 处理测试数据，可能没有标签
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                test_data, _ = batch_data  # 有标签的情况
            else:
                test_data = batch_data  # 无标签的情况
                
            test_data = test_data.to(device)
            
            # 获取当前批次的ID范围
            start_id = 501 + batch_idx * test_loader.batch_size
            end_id = min(start_id + len(test_data), 501 + len(test_dataset))
            batch_ids = list(range(start_id, end_id))
            
            # 前向传播
            _, logits, _ = model(test_data)
            
            # 计算负样本的似然值 (使用softmax后的概率)
            probs = F.softmax(logits, dim=1)
            negative_probs = probs[:, 0]  # 负样本(类别0)的概率
            
            # 收集结果
            test_ids.extend(batch_ids)
            negative_likelihoods.extend(negative_probs.cpu().numpy())
    
    # 创建DataFrame
    results_df = pd.DataFrame({
        'id': test_ids,
        'negative_likelihood': negative_likelihoods
    })
    
    # 按照负样本似然值排序（从高到低）
    results_df = results_df.sort_values('negative_likelihood', ascending=False).reset_index(drop=True)
    
    # 根据排序结果分配标签
    # 前50%标记为负样本(0)，后50%标记为正样本(1)
    total_samples = len(results_df)
    threshold_index = int(total_samples * 0.5)
    
    results_df['target'] = 0  # 默认为负样本
    results_df.loc[threshold_index:, 'target'] = 1  # 后50%标记为正样本
    
    # 按照ID重新排序，恢复原始顺序
    results_df = results_df.sort_values('id').reset_index(drop=True)
    
    # 只保留id和target列，按照提交样例格式
    submission_df = results_df[['id', 'target']]
    
    # 保存结果
    submission_df.to_csv(output_csv_path, index=False)
    print(f"预测结果已保存到: {output_csv_path}")
    
    # 打印统计信息
    positive_count = submission_df['target'].sum()
    negative_count = len(submission_df) - positive_count
    print(f"预测结果统计:")
    print(f"正样本数量: {positive_count} ({positive_count/len(submission_df)*100:.2f}%)")
    print(f"负样本数量: {negative_count} ({negative_count/len(submission_df)*100:.2f}%)")
    
    return submission_df

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型路径 - 使用最新训练的模型
    model_path = "model/aec_model_0.9800.pth"
    
    # 测试数据路径
    test_csv_path = "data/test.csv"
    
    # 输出路径
    output_csv_path = "submission.csv"
    
    # 运行预测
    submission_df = predict_test_data(model_path, test_csv_path, output_csv_path, device)
    
    # 显示前10行预测结果
    print("\n前10行预测结果:")
    print(submission_df.head(10))