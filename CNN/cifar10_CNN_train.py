import argparse
import os

import yaml

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from model import Qiming_2, Qiming_3_1, Qiming_3_2

# ---------- 命令行参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description='Simple CNN Training on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results (loss curve, confusion matrix, model)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


# ---------- 训练一个 epoch ----------
def train_epoch(model, loader, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / len(loader)


# ---------- 验证 ----------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, epoch, epochs):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [Eval ]', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, np.concatenate(all_preds), np.concatenate(all_targets)


# ---------- 详细测试评估（Top-1、Top-5、F1）----------
@torch.no_grad()
def detailed_test_evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='Test  ', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

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
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    macro_f1 = f1_score(targets_all, preds_all, average='macro')
    return avg_loss, top1_acc, top5_acc, macro_f1, preds_all, targets_all


# ---------- 绘图 ----------
def plot_training_curves(train_losses, val_losses, lr_history, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上方：损失曲线
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 下方：学习率曲线
    ax2.plot(lr_history, label='Learning Rate', marker='^', color='C2')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Training curves saved to {save_path}')


def plot_confusion_matrix(all_targets, all_preds, save_path):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Confusion matrix saved to {save_path}')


# ---------- 主流程 ----------
def main():
    args = parse_args()

    print(f'[Config] device={args.device}  batch_size={args.batch_size}  '
          f'epochs={args.epochs}  lr={args.lr}  save_dir={args.save_dir}')

    os.makedirs(args.save_dir, exist_ok=True)

    # 保存超参数配置到 YAML
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'device': args.device,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
    }
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f'[INFO] Config saved to {os.path.join(args.save_dir, "config.yaml")}')

    # 数据集
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                              std=(0.2470, 0.2435, 0.2616))  # CIFAR-10 标准归一化
    ])

    # 验证集/测试集：纯标准化，无增强
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 带增强的训练集
    train_data = torchvision.datasets.CIFAR10(args.data_dir, train=True,
                                              transform=train_transform, download=True)
    # 不带增强的训练集
    train_data_copy = torchvision.datasets.CIFAR10(args.data_dir, train=True,
                                              transform=val_test_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(args.data_dir, train=False,
                                             transform=val_test_transform, download=True)

    # 从不带增强的训练集中分出 10000 张作为验证集
    train_size = len(train_data) - 10000
    val_size = 10000
    train_data, _ = random_split(train_data, [train_size, val_size])
    _, val_data = random_split(train_data_copy, [train_size, val_size])
    print(f'Train samples: {len(train_data)}  |  Val samples: {len(val_data)}  |  Test samples: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # shuffle 默认为 False
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 模型、损失函数、优化器
    model = Qiming_3_2().to(args.device)
    loss_fn = CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # 学习率调度器（每 10 个 epoch 下降一次）
    # 注意: step_size=10，若 --epochs 小于 10 则调度器不会生效
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        patience=4,
        factor=0.5
        )

    # 训练循环
    train_losses, val_losses, lr_history = [], [], []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, loss_fn, optimizer, args.device, epoch, args.epochs)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_fn, args.device, epoch, args.epochs)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        lr_history.append(scheduler.get_last_lr()[0])

        print(f'>>> Epoch {epoch + 1:2d}/{args.epochs}  |  '
              f'Train Loss: {tr_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'    [*] Best model saved (val_loss={val_loss:.4f})')

        scheduler.step(val_loss)

    # 保存训练曲线（损失 + 学习率）
    plot_training_curves(train_losses, val_losses, lr_history,
                         os.path.join(args.save_dir, 'training_curves.png'))

    # 加载验证集最优模型权重再进行测试
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    print(f'Loaded best model (val_loss={best_val_loss:.4f}) for test evaluation.')

    # 训练完成后，在测试集上做最终评估
    print('Evaluating on test set...')
    test_loss, top1_acc, top5_acc, macro_f1, test_preds, test_targets = \
        detailed_test_evaluate(model, test_loader, loss_fn, args.device)

    # 打印并保存指标
    metrics_str = (
        '========== Test Set Metrics ==========\n'
        f'Test Loss:     {test_loss:.4f}\n'
        f'Top-1 Acc:     {top1_acc:.4f}  ({top1_acc * 100:.2f}%)\n'
        f'Top-5 Acc:     {top5_acc:.4f}  ({top5_acc * 100:.2f}%)\n'
        f'Macro-F1:      {macro_f1:.4f}\n'
        '======================================\n'
    )
    print(metrics_str)
    with open(os.path.join(args.save_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(metrics_str)
    print(f'[INFO] Test metrics saved to {os.path.join(args.save_dir, "test_metrics.txt")}')

    plot_confusion_matrix(test_targets, test_preds,
                          os.path.join(args.save_dir, 'confusion_matrix.png'))

    print('Done!')


if __name__ == '__main__':
    main()
