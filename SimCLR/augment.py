import torchvision.transforms as T

class SimCLRTrsform:
    def __init__(self, size=96):
        # 全套强增强
        self.transform = T.Compose([
            T.RandomResizedCrop(size), # 随机截取一块区域，再放大到原来尺寸
            T.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2), # 随机色彩抖动
            T.RandomGrayscale(p=0.2), # 随机灰度化
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)), # 高斯模糊
            T.ToTensor(),
            T.Normalize(mean=[0.447, 0.440, 0.407], 
                        std=[0.260, 0.257, 0.271])
        ])

    def __call__(self, img):
        # 返回两个不同的视图
        img_v1 = self.transform(img)
        img_v2 = self.transform(img)
        return img_v1, img_v2
    
def get_linear_transform(size=96):
    # 线性弱评估增强
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.447, 0.440, 0.407], 
                    std=[0.260, 0.257, 0.271])
    ])
