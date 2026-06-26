from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from augment import SimCLRTrsform, get_linear_transform

# 自监督预训练数据集（无标签）
class STL10ContrastDataset(Dataset):
    def __init__(self, root, split='unlabeled'):
        self.data = datasets.STL10(root=root, split=split, download=True)
        self.aug = SimCLRTrsform()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, _ = self.data[idx]
        img_v1, img_v2 = self.aug(img)
        return img_v1, img_v2

# 线性评估数据集（有标签）
class STL10LinearDataset(Dataset):
    def __init__(self, root, split='train'):
        self.data = datasets.STL10(root=root, split=split, download=True)
        self.transform = get_linear_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label
    
# 生成 dataloder
def get_train_dataloders(batch_size, root='./Dataset/STL10', num_workers=0):
    print('数据集路径：', root)

    train_contrast = STL10ContrastDataset(root=root, split='unlabeled')
    length = len(train_contrast)
    train_loader = DataLoader(train_contrast, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    print(f'成功加载无标签数据集，样本数：{length}')

    return train_loader

def get_linear_dataloders(batch_size, root='./Dataset/STL10', num_workers=0):
    print('数据集路径：', root)
    linear_train = STL10LinearDataset(root=root, split='train')
    linear_train_dataloder = DataLoader(linear_train, batch_size, shuffle=True, num_workers=num_workers)
    linear_test = STL10LinearDataset(root=root, split='test')
    linear_test_dataloder = DataLoader(linear_test, batch_size, shuffle=False, num_workers=num_workers)
    print('成功加载带标签数据集')
    return linear_train_dataloder, linear_test_dataloder
