import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rand_bbox(size, lam):
    _, _, height, width = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    # 随机选择替换区域中心点，并裁剪到图像边界内。
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(imgs, targets, alpha):
    # CutMix 从 Beta(alpha, alpha) 采样混合比例，再把另一张图的局部区域贴到当前图上。
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(imgs.size(0), device=imgs.device)
    target_a = targets
    target_b = targets[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    mixed_imgs = imgs.clone()
    mixed_imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[rand_index, :, bby1:bby2, bbx1:bbx2]

    # 边界裁剪后，实际贴图面积可能与采样的 lam 不完全一致，因此按真实面积重新计算标签权重。
    patch_area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1.0 - patch_area / (imgs.size(-1) * imgs.size(-2))
    return mixed_imgs, target_a, target_b, lam

# ---------- 绘图 ----------
def plot_training_curves(train_losses, train_losses_ce, train_losses_supcon,
                        val_losses, val_losses_ce, val_losses_supcon,
                        lr_history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    epochs = range(1, len(train_losses) + 1)

    # 图1：总损失曲线
    axes[0,0].plot(epochs, train_losses, label='Train Total Loss', marker='o', markersize=3, color='C0')
    axes[0,0].plot(epochs, val_losses, label='Val Total Loss', marker='s', markersize=3, color='C1')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training and Validation Total Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # 图2：CrossEntropy Loss 曲线
    axes[0,1].plot(epochs, train_losses_ce, label='Train CE Loss', marker='o', markersize=3, color='C0')
    axes[0,1].plot(epochs, val_losses_ce, label='Val CE Loss', marker='s', markersize=3, color='C1')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].set_title('Training and Validation CrossEntropy Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # 图3：SupCon Loss 曲线
    axes[1,0].plot(epochs, train_losses_supcon, label='Train SupCon Loss', marker='o', markersize=3, color='C0')
    axes[1,0].plot(epochs, val_losses_supcon, label='Val SupCon Loss', marker='s', markersize=3, color='C1')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_title('Training and Validation SupCon Loss')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # 图4：学习率曲线
    axes[1,1].plot(epochs, lr_history, label='Learning Rate', marker='^', color='C2', markersize=3)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Training curves saved to {save_path}')


def plot_confusion_matrix(all_targets, all_preds, save_path):
    # 加载 CIFAR-100 超类标签名
    import pickle
    meta_path = os.path.join('Dataset', 'cifar-100-python', 'meta')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    coarse_names = [name.decode() for name in meta[b'coarse_label_names']]

    # 构建 fine → coarse 映射（每个细类 0-99 → 对应超类 0-19）
    train_batch_path = os.path.join('Dataset', 'cifar-100-python', 'train')
    with open(train_batch_path, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    fine_to_coarse = {}
    for fine, coarse in zip(train_data[b'fine_labels'], train_data[b'coarse_labels']):
        if fine not in fine_to_coarse:
            fine_to_coarse[fine] = coarse

    # 将 100 类的预测/标签映射到 20 超类
    coarse_targets = np.array([fine_to_coarse[t] for t in all_targets])
    coarse_preds = np.array([fine_to_coarse[p] for p in all_preds])

    cm = confusion_matrix(coarse_targets, coarse_preds)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=coarse_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix (20 Superclasses)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Confusion matrix saved to {save_path}')

def save_config(args):
        # 保存超参数配置到 YAML
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'warmup_epochs': args.warmup_epochs,
        'alpha': args.alpha,
        'cutmix_prob': args.cutmix_prob,
        'cutmix_alpha': args.cutmix_alpha,
        'device': args.device,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
    }
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f'[INFO] Config saved to {os.path.join(args.save_dir, "config.yaml")}')
