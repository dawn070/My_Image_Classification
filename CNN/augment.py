import torchvision.transforms as T

class CIFAR100_Transform:
    def __init__(self, split='train'):
        if split == 'train':
            # 训练集：引入 RandAugment + Cutout 效果的增强组合
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=9), # 引入 RandAugment
                T.ToTensor(),
                T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2)) # 引入 Cutout 效果
            ])
        
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ])

    def __call__(self, img):
        img = self.transform(img)
        return img
