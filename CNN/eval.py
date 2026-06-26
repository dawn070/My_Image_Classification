import os
import argparse
import pickle
import warnings

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import Qiming_4_5
from augment import CIFAR100_Transform
from loss import SupConLoss
from torch.nn import CrossEntropyLoss

warnings.filterwarnings('ignore')


def load_fine_to_coarse_mapping(data_dir):
    """加载CIFAR-100的细类到超类的映射关系"""
    meta_path = os.path.join(data_dir, 'cifar-100-python', 'meta')
    train_batch_path = os.path.join(data_dir, 'cifar-100-python', 'train')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    coarse_names = [name.decode() for name in meta[b'coarse_label_names']]
    fine_names = [name.decode() for name in meta[b'fine_label_names']]

    with open(train_batch_path, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    fine_to_coarse = {}
    for fine, coarse in zip(train_data[b'fine_labels'], train_data[b'coarse_labels']):
        if fine not in fine_to_coarse:
            fine_to_coarse[fine] = coarse

    # 也可以获取coarse到fine的映射（用于子类别标签）
    coarse_to_fines = {i: [] for i in range(20)}
    for fine, coarse in fine_to_coarse.items():
        coarse_to_fines[coarse].append(fine)

    return coarse_names, fine_names, fine_to_coarse, coarse_to_fines


@torch.no_grad()
def evaluate(model, loader, device, loss_fn_ce, loss_fn_supcon, alpha):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='Evaluating', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs, emb = model(imgs)

        loss_ce = loss_fn_ce(outputs, targets)
        loss_supcon = loss_fn_supcon(emb, targets)
        loss = loss_ce + alpha * loss_supcon
        total_loss += loss.item()

        # Top-1
        preds = outputs.argmax(1)
        top1_correct += (preds == targets).sum().item()

        # Top-5
        _, top5_idx = outputs.topk(5, dim=1)
        top5_correct += (top5_idx == targets.view(-1, 1)).any(dim=1).sum().item()

        total += targets.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)

    return avg_loss, top1_acc, top5_acc, preds_all, targets_all


def plot_single_superclass_cm(cm, superclass_idx, superclass_name, fine_labels, save_path):
    """为单个超类绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fine_labels)
    disp.plot(cmap='Blues', values_format='d', ax=ax, xticks_rotation=45)

    # 计算准确率
    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0

    ax.set_title(f'Superclass {superclass_idx}: {superclass_name}\nAccuracy: {acc:.4f} (Total samples: {np.sum(cm)})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-100 model and plot confusion matrices per superclass')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='Dataset', help='Dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./eval_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.1, help='SupConLoss的权重')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cuda or cpu')
    args = parser.parse_args()

    print(f'[Config] checkpoint={args.checkpoint}  device={args.device}  '
          f'batch_size={args.batch_size}  alpha={args.alpha}')

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载细类到超类的映射
    coarse_names, fine_names, fine_to_coarse, coarse_to_fines = load_fine_to_coarse_mapping(args.data_dir)
    print(f'Loaded {len(coarse_names)} superclasses and {len(fine_names)} fine classes')

    # 加载测试集
    test_data = torchvision.datasets.CIFAR100(
        args.data_dir, train=False,
        transform=CIFAR100_Transform(split='test'),
        download=True
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print(f'Test samples: {len(test_data)}')

    # 加载模型
    device = torch.device(args.device)
    model = Qiming_4_5().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f'Loaded model from {args.checkpoint}')

    # 损失函数
    loss_fn_ce = CrossEntropyLoss(label_smoothing=0.1).to(device)
    loss_fn_supcon = SupConLoss(temperature=0.07)

    # 评估
    test_loss, top1_acc, top5_acc, preds_all, targets_all = evaluate(
        model, test_loader, device, loss_fn_ce, loss_fn_supcon, args.alpha
    )

    # 打印整体指标
    metrics_str = (
        '========== Test Set Metrics ==========\n'
        f'Test Loss:     {test_loss:.4f}\n'
        f'Top-1 Acc:     {top1_acc:.4f}  ({top1_acc * 100:.2f}%)\n'
        f'Top-5 Acc:     {top5_acc:.4f}  ({top5_acc * 100:.2f}%)\n'
        '======================================\n'
    )
    print(metrics_str)
    with open(os.path.join(args.save_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(metrics_str)

    # 绘制整体混淆矩阵（20超类）
    coarse_targets = np.array([fine_to_coarse[t] for t in targets_all])
    coarse_preds = np.array([fine_to_coarse[p] for p in preds_all])
    cm_coarse = confusion_matrix(coarse_targets, coarse_preds)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_coarse, display_labels=coarse_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix (20 Superclasses)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrix_superclass.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved overall confusion matrix to {os.path.join(args.save_dir, "confusion_matrix_superclass.png")}')

    # 为每个超类绘制混淆矩阵（显示其内部的细类）
    for superclass_idx in range(20):
        fine_indices = coarse_to_fines[superclass_idx]
        fine_labels = [fine_names[i] for i in fine_indices]

        # 获取属于该超类的样本
        mask = np.isin(targets_all, fine_indices)
        if np.sum(mask) == 0:
            print(f'[WARNING] No samples found for superclass {superclass_idx} ({coarse_names[superclass_idx]})')
            continue

        # 只保留属于该超类的样本（预测也只考虑该超类内部的细类）
        # 如果预测不在这个超类内部，则标记为"-1"（在混淆矩阵中忽略或单独列）
        filtered_preds = []
        filtered_targets = []

        for target, pred in zip(targets_all[mask], preds_all[mask]):
            if pred in fine_indices:
                # 将预测和标签映射到0~(n-1)的索引
                pred_idx = fine_indices.index(pred)
                target_idx = fine_indices.index(target)
                filtered_preds.append(pred_idx)
                filtered_targets.append(target_idx)
            else:
                # 预测为其他超类，标记为-1
                filtered_preds.append(len(fine_indices))  # 最后一行
                filtered_targets.append(fine_indices.index(target))

        # 添加"Other"标签
        display_labels = fine_labels + ['Other']

        cm = confusion_matrix(filtered_targets, filtered_preds, labels=list(range(len(display_labels))))

        # 保存单个超类的混淆矩阵
        save_path = os.path.join(args.save_dir, f'confusion_matrix_superclass_{superclass_idx:02d}_{coarse_names[superclass_idx]}.png')
        plot_single_superclass_cm(cm, superclass_idx, coarse_names[superclass_idx], display_labels, save_path)

        # 计算该超类的准确率
        acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        print(f'Superclass {superclass_idx:2d} ({coarse_names[superclass_idx]}): Acc={acc:.4f}, Samples={np.sum(mask)}')

    print(f'\nAll confusion matrices saved to {args.save_dir}')
    print('Done!')


if __name__ == '__main__':
    main()