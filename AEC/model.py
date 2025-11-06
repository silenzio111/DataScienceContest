from matplotlib import scale
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    变分自编码器(VAE)用于降维
    针对小数据集(500样本)设计，参数量适中避免过拟合
    """
    def __init__(self,input_dim=22, latent_dim=5,hidden_dims=[14,8,6]):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器网络
        encoder_layers = []
        prev_dim = input_dim
        
        # 构建编码器隐藏层
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )
            encoder_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
            encoder_layers.append(
                nn.LeakyReLU(0.2)
            )
            encoder_layers.append(
                nn.Dropout(0.2)  # 添加dropout防止过拟合
            )
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和方差层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # 解码器网络
        decoder_layers = []
        prev_dim = latent_dim
        
        # 构建解码器隐藏层（反向结构）
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )
            decoder_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
            decoder_layers.append(
                nn.LeakyReLU(0.2)
            )
            decoder_layers.append(
                nn.Dropout(0.2)
            )
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, input_dim)
    
    def encode(self, x):
        """编码器，返回均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """解码器"""
        h = self.decoder(z)
        return self.output_layer(h)
    
    def forward(self, x):
        """前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def get_embedding(self, x):
        """获取降维后的嵌入表示"""
        mu, _ = self.encode(x)
        return mu
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0,label=None):
        """
        VAE损失函数：重构损失 + KL散度
        
        参数:
            recon_x: 重构数据
            x: 原始数据
            mu: 均值
            log_var: 对数方差
            beta: KL散度的权重系数
        """
        sacle=5
        # 重构损失 (均方误差)
        recon_loss = get_per_sample_recon_loss(recon_x, x)*((-2*(label-0.5))*sacle+scale-1)
        recon_loss=torch.sum(recon_loss)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
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
# 测试模型
if __name__ == "__main__":
    # 创建VAE模型实例
    vae_model = VAE(input_dim=22, latent_dim=10)
    
    # 打印VAE模型结构
    print("VAE模型结构:")
    print(vae_model)
    
    # 打印VAE参数量
    vae_param_count = vae_model.count_parameters()
    print(f"\nVAE模型参数量: {vae_param_count}")
    
    # 测试VAE前向传播
    batch_size = 32
    x = torch.randn(batch_size, 22)
    
    # VAE前向传播
    recon_x, mu, log_var = vae_model(x)
    
    # 计算VAE损失
    total_loss, recon_loss, kl_loss = vae_model.loss_function(recon_x, x, mu, log_var)
    
    print(f"\nVAE测试结果:")
    print(f"输入形状: {x.shape}")
    print(f"重构输出形状: {recon_x.shape}")
    print(f"潜在表示形状: {mu.shape}")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"重构损失: {recon_loss.item():.4f}")
    print(f"KL散度损失: {kl_loss.item():.4f}")
    
    # 测试VAE降维功能
    embedding = vae_model.get_embedding(x)
    print(f"VAE降维后形状: {embedding.shape}")
    
    # 创建AEC模型实例
    aec_model = AEC(input_dim=22, latent_dim=10)
    
    # 打印AEC模型结构
    print("\n\nAEC模型结构:")
    print(aec_model)
    
    # 打印AEC参数量
    aec_param_count = aec_model.count_parameters()
    print(f"\nAEC模型参数量: {aec_param_count}")
    print(f"AEC与VAE参数量比较: {aec_param_count/vae_param_count:.2f}x")
    
    # 测试AEC前向传播
    labels = torch.randint(0, 2, (batch_size,))
    
    # AEC前向传播
    recon_x_aec, logits, z = aec_model(x)
    
    # 计算AEC损失
    total_loss_aec, recon_loss_aec, cls_loss = aec_model.loss_function(recon_x_aec, x, logits, labels)
    
    print(f"\nAEC测试结果:")
    print(f"输入形状: {x.shape}")
    print(f"重构输出形状: {recon_x_aec.shape}")
    print(f"分类输出形状: {logits.shape}")
    print(f"潜在表示形状: {z.shape}")
    print(f"总损失: {total_loss_aec.item():.4f}")
    print(f"重构损失: {recon_loss_aec.item():.4f}")
    print(f"分类损失: {cls_loss.item():.4f}")
    
    # 测试AEC降维功能
    embedding_aec = aec_model.get_embedding(x)
    print(f"AEC降维后形状: {embedding_aec.shape}")
    
    # 测试分类器单独使用
    classifier_output = aec_model.classify(z)
    print(f"分类器输出形状: {classifier_output.shape}")
    
    # 测试预测概率
    probs = F.softmax(classifier_output, dim=1)
    print(f"分类概率形状: {probs.shape}")
    print(f"预测类别: {torch.argmax(probs, dim=1)}")


class AEC(nn.Module):
    """
    自编码器分类器(AEC)结合了自编码器和分类器
    包含三个部分：encoder、decoder和classifier
    参照VAE的参数量设计，避免过拟合
    """
    def __init__(self, input_dim=22, latent_dim=5, hidden_dims=[14,8,6], num_classes=2):
        super(AEC, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 编码器网络 (与VAE相同的结构)
        encoder_layers = []
        prev_dim = input_dim
        
        # 构建编码器隐藏层
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )
            encoder_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
            encoder_layers.append(
                nn.LeakyReLU(0.2)
            )
            encoder_layers.append(
                nn.Dropout(0.2)  # 添加dropout防止过拟合
            )
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在表示层
        self.latent_layer = nn.Linear(prev_dim, latent_dim)
        
        # 解码器网络 (与VAE相同的结构)
        decoder_layers = []
        prev_dim = latent_dim
        
        # 构建解码器隐藏层（反向结构）
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )
            decoder_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
            decoder_layers.append(
                nn.LeakyReLU(0.2)
            )
            decoder_layers.append(
                nn.Dropout(0.2)
            )
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, input_dim)
        
        # 分类器网络
        classifier_layers = []
        prev_dim = latent_dim
        
        # 构建分类器隐藏层
        for hidden_dim in [8, 6]:  # 使用较小的隐藏层避免过拟合
            classifier_layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )
            classifier_layers.append(
                nn.BatchNorm1d(hidden_dim)
            )
            classifier_layers.append(
                nn.LeakyReLU(0.2)
            )
            classifier_layers.append(
                nn.Dropout(0.2)
            )
            prev_dim = hidden_dim
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 分类器输出层
        self.classifier_output = nn.Linear(prev_dim, num_classes)
    
    def encode(self, x):
        """编码器，返回潜在表示"""
        h = self.encoder(x)
        z = self.latent_layer(h)
        return z
    
    def decode(self, z):
        """解码器"""
        h = self.decoder(z)
        return self.output_layer(h)
    
    def classify(self, z):
        """分类器，基于潜在表示进行分类"""
        h = self.classifier(z)
        return self.classifier_output(h)
    
    def forward(self, x):
        """前向传播"""
        # 编码
        z = self.encode(x)
        
        # 解码
        x_recon = self.decode(z)
        
        # 分类
        logits = self.classify(z)
        
        return x_recon, logits, z
    
    def get_embedding(self, x):
        """获取降维后的嵌入表示"""
        return self.encode(x)
    
    def loss_function(self, recon_x, x, logits, labels, alpha=1.0, beta=1.0):
        """
        AEC损失函数：重构损失 + 分类损失
        
        参数:
            recon_x: 重构数据
            x: 原始数据
            logits: 分类器输出
            labels: 真实标签
            alpha: 重构损失的权重系数
            beta: 分类损失的权重系数
        """
        # 重构损失 (均方误差)
        recon_loss = F.mse_loss(recon_x, x)
        
        # 分类损失 (交叉熵)
        cls_loss = F.cross_entropy(logits, labels)
        
        return alpha * recon_loss + beta * cls_loss, recon_loss, cls_loss
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)