from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, AdaptiveAvgPool2d
from torch.nn import Sigmoid, ReLU, BatchNorm1d, BatchNorm2d, Dropout, GELU

# 基础十分类网络结构
class Qiming_1(nn.Module):
    def __init__(self):
        super(Qiming_1, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(in_features=64 * 4 * 4, out_features=64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
# 添加加激活函数 | BN | Dropout
class Qiming_2(nn.Module):
    def __init__(self):
        super(Qiming_2, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            BatchNorm2d(32),
            # Sigmoid(),  使用这个激活函数出现梯度消失现象
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 32, 5, 1, 2),
            BatchNorm2d(32),
            # Sigmoid(),
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 64, 5, 1, 2),
            BatchNorm2d(64),
            # Sigmoid(),
            ReLU(),
            MaxPool2d(2),

            Flatten(),
            Dropout(p=0.4),
            Linear(in_features=64 * 4 * 4, out_features=64),

            ReLU(),  # 全连接层也要加激活函数
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 5×5改为3×3卷积
class Qiming_3_1(nn.Module):
    def __init__(self):
        super(Qiming_3_1, self).__init__()
        self.model = nn.Sequential(  # input: 32*32*3
            Conv2d(3, 32, 3, 1, 1),  # 32*32*32
            Conv2d(32, 32, 3, 1, 1),  # 32*32*32
            BatchNorm2d(32),
            GELU(),
            MaxPool2d(2, 2),  # 16*16*32

            Conv2d(32, 64, 3, 1, 1), # 16*16*64
            Conv2d(64, 64, 3, 1, 1), # 16*16*64
            BatchNorm2d(64),
            GELU(),
            MaxPool2d(2, 2),  # 8*8*64

            Conv2d(64, 128, 3, 1, 1), # 8*8*128
            Conv2d(128, 128, 3, 1, 1), # 8*8*128
            Conv2d(128, 128, 3, 1, 1), # 8*8*128
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(2, 2),  # 4*4*128

            Flatten(), # # 1*1*N
            Dropout(0.4),
            Linear(in_features=4*4*128, out_features=128), # 1*1*128
            GELU(),
            Linear(in_features=128, out_features=10) # 1*1*10
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# 自己手写的卷积类，包含 卷积 + BN + GELU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = BatchNorm2d(out_channels)
        self.gelu = GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

# 简化写法，并保证每个卷积输出都有 NB 和 activation
class Qiming_3_2(nn.Module):
    def __init__(self):
        super(Qiming_3_2, self).__init__()
        self.model = nn.Sequential(  # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 32, 3, 1, 1),  # 32*32*32
            MaxPool2d(2, 2),  # 16*16*32

            ConvBlock(32, 64, 3, 1, 1), # 16*16*64
            ConvBlock(64, 64, 3, 1, 1), # 16*16*64
            MaxPool2d(2, 2),  # 8*8*64

            ConvBlock(64, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            MaxPool2d(2, 2),  # 4*4*128

            Flatten(), # # 1*1*N
            Linear(in_features=4*4*128, out_features=512), # 1*1*512
            BatchNorm1d(512),
            GELU(),
            Dropout(0.4),

            Linear(in_features=512, out_features=256), # 1*1*256
            BatchNorm1d(256),
            GELU(),
            Dropout(0.4),

            Linear(in_features=256, out_features=100) # 1*1*100
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

# 增加网络深度和特征维度
class Qiming_3_3(nn.Module):
    def __init__(self):
        super(Qiming_3_3, self).__init__()
        self.model = nn.Sequential(  # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 32, 3, 1, 1),  # 32*32*32
            MaxPool2d(2, 2),  # 16*16*32

            ConvBlock(32, 64, 3, 1, 1), # 16*16*64
            ConvBlock(64, 64, 3, 1, 1), # 16*16*64
            MaxPool2d(2, 2),  # 8*8*64

            ConvBlock(64, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            MaxPool2d(2, 2),  # 4*4*128

            ConvBlock(128, 256, 3, 1, 1), # 4*4*256
            ConvBlock(256, 256, 3, 1, 1), # 4*4*256
            ConvBlock(256, 256, 3, 1, 1), # 4*4*256
            MaxPool2d(2, 2), # 2*2*256

            Flatten(), # # 1*1*N
            Linear(in_features=2*2*256, out_features=512), # 1*1*512
            BatchNorm1d(512),
            GELU(),
            Dropout(0.4),

            Linear(in_features=512, out_features=256), # 1*1*256
            BatchNorm1d(256),
            GELU(),
            Dropout(0.4),

            Linear(in_features=256, out_features=100) # 1*1*100
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# FC改为全局池化
class Qiming_3_4(nn.Module):
    def __init__(self):
        super(Qiming_3_4, self).__init__()
        self.model = nn.Sequential(  # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 32, 3, 1, 1),  # 32*32*32
            MaxPool2d(2, 2),  # 16*16*32

            ConvBlock(32, 64, 3, 1, 1), # 16*16*64
            ConvBlock(64, 64, 3, 1, 1), # 16*16*64
            MaxPool2d(2, 2),  # 8*8*64

            ConvBlock(64, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            ConvBlock(128, 128, 3, 1, 1), # 8*8*128
            MaxPool2d(2, 2),  # 4*4*128

            ConvBlock(128, 256, 3, 1, 1), # 4*4*256
            ConvBlock(256, 256, 3, 1, 1), # 4*4*256
            ConvBlock(256, 256, 3, 1, 1), # 4*4*256
            MaxPool2d(2, 2), # 2*2*256

            AdaptiveAvgPool2d(1), # 1*1*256
            Flatten(),
            Dropout(0.3),
            Linear(256, 100)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
        # 即使卷积形式相同，也必须定义两个 conv 和 bn，否则两层卷积的权重共享！
        self.conv1 = Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = BatchNorm2d(channels)
        self.act = GELU()
        self.conv2 = Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = BatchNorm2d(channels)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # 关键部分：残差连接

        out = self.act(out)

        return out
    
# 使用残差连接，加深网络
class Qiming_4_1(nn.Module):
    def __init__(self):
        super(Qiming_4_1, self).__init__()
        self.model = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64
            MaxPool2d(2, 2),  # 16*16*64

            ResidualBlock(64),  # 16*16*64
            ConvBlock(64, 128, 3, 1, 1), # 16*16*128
            MaxPool2d(2, 2), # 8*8*128

            ResidualBlock(128), # 8*8*128
            ConvBlock(128, 256, 3, 1, 1), # 8*8*256
            MaxPool2d(2, 2), # 4*4*256

            ResidualBlock(256), # 4*4*256
            ConvBlock(256, 512, 3, 1, 1), # 4*4*512
            MaxPool2d(2, 2), # 2*2*512

            ResidualBlock(512), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten(), # 512
            Dropout(0.4),
            Linear(512, 100) # 100

        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# 取消最大池化，改用步长为2的卷积核实现下采样
class Qiming_4_2(nn.Module):
    def __init__(self):
        super(Qiming_4_2, self).__init__()
        self.model = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64
            ConvBlock(64, 64, 3, 2, 1), # 16*16*64

            ResidualBlock(64),  # 16*16*64
            ConvBlock(64, 128, 3, 1, 1), # 16*16*128
            ConvBlock(128, 128, 3, 2, 1), # 8*8*128

            ResidualBlock(128), # 8*8*128
            ConvBlock(128, 256, 3, 1, 1), # 8*8*256
            ConvBlock(256, 256, 3, 2, 1), # 4*4*256

            ResidualBlock(256), # 4*4*256
            ConvBlock(256, 512, 3, 1, 1), # 4*4*512
            ConvBlock(512, 512, 3, 2, 1), # 2*2*512

            ResidualBlock(512), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten(), # 512
            Dropout(0.4),
            Linear(512, 100) # 100

        )
    
    def forward(self, x):
        x = self.model(x)
        return x

# 进一步加深网络
class Qiming_4_3(nn.Module):
    def __init__(self):
        super(Qiming_4_3, self).__init__()
        self.model = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64
            ConvBlock(64, 64, 3, 2, 1), # 16*16*64

            ResidualBlock(64),  # 16*16*64
            ResidualBlock(64),  # 16*16*64
            ConvBlock(64, 128, 3, 1, 1), # 16*16*128
            ConvBlock(128, 128, 3, 2, 1), # 8*8*128

            ResidualBlock(128), # 8*8*128
            ResidualBlock(128), # 8*8*128
            ConvBlock(128, 256, 3, 1, 1), # 8*8*256
            ConvBlock(256, 256, 3, 2, 1), # 4*4*256

            ResidualBlock(256), # 4*4*256
            ResidualBlock(256), # 4*4*256
            ConvBlock(256, 512, 3, 1, 1), # 4*4*512
            ConvBlock(512, 512, 3, 2, 1), # 2*2*512

            ResidualBlock(512), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten(), # 512
            Dropout(0.4),
            Linear(512, 100) # 100

        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# 前两层用高分辨率
class Qiming_4_4(nn.Module):
    def __init__(self):
        super(Qiming_4_4, self).__init__()
        self.model = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64

            ResidualBlock(64),  # 32*32*64
            ConvBlock(64, 128, 3, 1, 1), # 32*32*128
            ConvBlock(128, 128, 3, 2, 1), # 16*16*128

            ResidualBlock(128), # 16*16*128
            ConvBlock(128, 256, 3, 1, 1), # 16*16*256
            ConvBlock(256, 256, 3, 2, 1), # 8*8*256

            ResidualBlock(256), # 8*8*256
            ConvBlock(256, 512, 3, 1, 1), # 8*8*512
            ConvBlock(512, 512, 3, 2, 1), # 4*4*512

            ResidualBlock(512), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten(), # 512
            Dropout(0.4),
            Linear(512, 100) # 100
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
# 增加 project-head 用于 Supervised Contrastive Loss
class Qiming_4_5(nn.Module):
    def __init__(self):
        super(Qiming_4_5, self).__init__()
        # 骨干网络
        self.backbone = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64

            ResidualBlock(64),  # 32*32*64
            ConvBlock(64, 128, 3, 1, 1), # 32*32*128
            ConvBlock(128, 128, 3, 2, 1), # 16*16*128

            ResidualBlock(128), # 16*16*128
            ConvBlock(128, 256, 3, 1, 1), # 16*16*256
            ConvBlock(256, 256, 3, 2, 1), # 8*8*256

            ResidualBlock(256), # 8*8*256
            ConvBlock(256, 512, 3, 1, 1), # 8*8*512
            ConvBlock(512, 512, 3, 2, 1), # 4*4*512

            ResidualBlock(512), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten() # 512
        )

        self.project_head = nn.Sequential(# input: 32*32*512
            Linear(512, 256),
            BatchNorm1d(256),
            GELU()
        )

        self.classifier = nn.Sequential(
            Dropout(0.4),
            Linear(256, 100)
            )

    def forward(self, x):

        feature = self.backbone(x)

        project = self.project_head(feature)

        logits = self.classifier(project)

        return logits, project

# SE注意力机制（Squeeze-and-Excitation）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            Linear(channels, channels//reduction),
            GELU(),
            Linear(channels//reduction, channels),
            Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # squeeze
        y = self.pool(x)
        y = y.view(b, c)

        # extraction
        y = self.fc(y)

        # reshape
        y = y.view(b, c, 1, 1)

        return x*y

# 加入SE的残差块
class SEResidualBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
    
        self.conv1 = Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = BatchNorm2d(channels)
        self.act = GELU()
        self.conv2 = Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = BatchNorm2d(channels)

        self.se = SEBlock(channels, reduction)

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out = out + identity  # 关键部分：残差连接

        out = self.act(out)

        return out

# 使用带SE的残差网络
class Qiming_5_1(nn.Module):
    def __init__(self):
        super(Qiming_5_1, self).__init__()
        self.model = nn.Sequential( # input: 32*32*3
            ConvBlock(3, 32, 3, 1, 1),   # 32*32*32
            ConvBlock(32, 64, 3, 1, 1), # 32*32*64

            SEResidualBlock(channels=64, reduction=8),  # 32*32*64
            ConvBlock(64, 128, 3, 1, 1), # 32*32*128
            ConvBlock(128, 128, 3, 2, 1), # 16*16*128

            SEResidualBlock(channels=64, reduction=8), # 16*16*128
            ConvBlock(128, 256, 3, 1, 1), # 16*16*256
            ConvBlock(256, 256, 3, 2, 1), # 8*8*256

            SEResidualBlock(channels=64, reduction=8), # 8*8*256
            ConvBlock(256, 512, 3, 1, 1), # 8*8*512
            ConvBlock(512, 512, 3, 2, 1), # 4*4*512

            SEResidualBlock(channels=64, reduction=8), # 2*2*512
            AdaptiveAvgPool2d(1), # 1*1*512

            Flatten(), # 512
            Dropout(0.2),
            Linear(512, 100) # 100
        )

    def forward(self, x):
        x = self.model(x)
        return x