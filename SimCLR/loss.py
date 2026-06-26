import torch
import torch.nn as nn
import torch.nn.functional as F

# 归一化温度交叉熵损失函数
class NTXenLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp

    def forward(self, z1, z2):
        B = z1.shape[0]  # 获取 batch

        z = torch.cat([z1, z2], dim=0)  # 拼接份不同的增强

        z = F.normalize(z, dim=1)  # 归一化

        similarity = torch.matmul(z, z.T)  # 计算相似矩阵
        similarity /= self.temp  # 除以温度系数

        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)  # 构造掩码：对角线

        similarity.masked_fill_(mask, -1e9)  # 将对角线相似度填充为极小值
        
        labels = torch.cat([
            torch.arange(B, 2*B),
            torch.arange(B)
        ]).to(z.device)  # 正样本标签

        loss = F.cross_entropy(similarity, labels, label_smoothing=0.1)

        return loss