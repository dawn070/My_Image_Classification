import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import SimCLR
from dataset import get_linear_dataloders
from utils import load_checkpoint
from tqdm import tqdm


def extract_features(model, data_loader, device):
    """提取特征和标签"""
    model.eval()
    features = []
    labels = []

    pbar = tqdm(data_loader, '正在提取特征 ', leave=True)

    with torch.no_grad():
        for imgs, lbls in pbar:
            imgs = imgs.to(device)
            feature, _ = model(imgs)  # 只使用backbone特征，不使用投影头
            features.append(feature.cpu().numpy())
            labels.append(lbls.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def plot_tsne(features, labels, save_path=None):
    """t-SNE可视化"""
    print('正在运行t-SNE降维...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    # STL-10类别名称
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer',
                   'dog', 'horse', 'monkey', 'ship', 'truck']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    plt.figure(figsize=(12, 10))

    for i in range(10):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=class_names[i], alpha=0.6, s=20)

    plt.legend(loc='best', fontsize=10)
    plt.title('t-SNE Visualization of SimCLR Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'可视化结果已保存到: {save_path}')

    plt.show()

def main():
    # 配置参数
    ckpt_path = 'SimCLR/runs/test03/simclr_checkpoints/checkpoint20.pth'  # 修改为你的权重路径
    data_dir = 'Dataset/STL10'
    save_dir = 'SimCLR/runs/test03'
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = SimCLR(out_dim=128).to(device)
    model = load_checkpoint(model, ckpt_path)
    # print(model)

    # 加载测试数据集（有标签）
    _, test_loader = get_linear_dataloders(batch_size=batch_size, root=data_dir)

    # 提取特征
    print('正在提取特征...')
    features, labels = extract_features(model, test_loader, device)
    print(f'特征形状: {features.shape}, 标签形状: {labels.shape}')

    # t-SNE可视化
    save_path = os.path.join(save_dir, 'tsne_visualization.png')
    plot_tsne(features, labels, save_path)

if __name__ == '__main__':
    main()