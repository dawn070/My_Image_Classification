import torch
import torchvision
import torchvision.transforms as transforms

# 基础图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# split可选：train / test / unlabeled / train+unlabeled
# download=True 自动下载；root指定存放目录
train_set = torchvision.datasets.STL10(
    root="dataset/STL10",
    split="train",
    download=False,
    transform=transform
)

test_set = torchvision.datasets.STL10(
    root="dataset/STL10",
    split="test",
    download=False,
    transform=transform
)

# 无标签数据集（10万张，半监督/自监督用）
unlabeled_set = torchvision.datasets.STL10(
    root="dataset/STL10",
    split="unlabeled",
    download=False,
    transform=transform
)

# 封装DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
print("训练集图片数量：", len(train_set))
print("单张图片尺寸：", train_set[0][0].shape)  # [3, 96, 96]