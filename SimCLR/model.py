import torch
import torch.nn as nn
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # 使用 ResNet18作为骨干网络(backbone)
        self.backbone = models.resnet18(weights=None)

        # 将resnet的第一层7*7卷积替换为3*3
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 相当于去掉这层
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 投影头（Project Head）
        self.project_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, out_dim)
        )
    
    def forward(self, x):
        feature = self.backbone(x)
        z = self.project_head(feature)
        return feature, z
    
# 线性分类头
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)