import argparse
import os
import warnings

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from loss import SupConLoss
from augment import CIFAR100_Transform
from utils import set_seed, seed_worker, cutmix_data, plot_training_curves, plot_confusion_matrix, save_config
from model import Qiming_2
from model import Qiming_3_1, Qiming_3_2, Qiming_3_3, Qiming_3_4
from model import Qiming_4_1, Qiming_4_2, Qiming_4_3, Qiming_4_4, Qiming_4_5
from model import Qiming_5_1

warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.*')

# ---------- 命令行参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description='Simple CNN Training on CIFAR-100')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热的轮次数（前 N 个 epoch 线性升温到 --lr），设为 0 则关闭预热')
    parser.add_argument('--alpha', type=float, default=0.1, help='SupConLoss的权重')
    parser.add_argument('--cutmix_prob', type=float, default=0.5,
                        help='CutMix 触发概率，设为 0 可关闭 CutMix')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix 的 Beta 分布参数，常用 1.0；值越大，混合区域比例越稳定')
    parser.add_argument('--data_dir', type=str, default='Dataset', help='Dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results (loss curve, confusion matrix, model)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes (0 = main process)')
    return parser.parse_args()


# ---------- 训练一个 epoch ----------
def train_epoch(model, loader, loss_fn_ce, loss_fn_supcon, alpha, optimizer, device, epoch, epochs, cutmix_prob=0.0, cutmix_alpha=1.0):
    model.train()
    total_loss = 0
    total_loss_ce = 0
    total_loss_supcon = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        # 仅训练阶段使用 CutMix；验证和测试保持原图输入，保证指标可比较。
        use_cutmix = cutmix_alpha > 0 and np.random.rand() < cutmix_prob
        if use_cutmix:
            imgs, targets_a, targets_b, lam = cutmix_data(imgs, targets, cutmix_alpha)
            outputs, emb = model(imgs)
            loss_ce = lam * loss_fn_ce(outputs, targets_a) + (1.0 - lam) * loss_fn_ce(outputs, targets_b)
            loss_supcon = lam * loss_fn_supcon(emb, targets_a) + (1.0 - lam) * loss_fn_supcon(emb, targets_b)
            loss = loss_ce + alpha*loss_supcon
        else:
            outputs, emb = model(imgs)
            loss_ce = loss_fn_ce(outputs, targets)
            loss_supcon = loss_fn_supcon(emb, targets)
            loss = loss_ce + alpha*loss_supcon

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_supcon += loss_supcon.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}', loss_ce=f'{loss_ce.item():.4f}', loss_supcon=f'{loss_supcon.item():.4f}')
    avg_loss = total_loss / len(loader)
    avg_loss_ce = total_loss_ce / len(loader)
    avg_loss_supcon = total_loss_supcon / len(loader)
    return avg_loss, avg_loss_ce, avg_loss_supcon


# ---------- 验证 ----------
@torch.no_grad()
def evaluate(model, loader, loss_fn_ce, loss_fn_supcon, alpha, device, epoch, epochs):
    model.eval()
    total_loss = 0
    total_loss_ce = 0
    total_loss_supcon = 0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [Eval ]', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs, emb = model(imgs)
        loss_ce = loss_fn_ce(outputs, targets)
        loss_supcon = loss_fn_supcon(emb, targets)
        loss = loss_ce + alpha*loss_supcon

        total_loss += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_supcon += loss_supcon.item()
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    avg_loss_ce = total_loss_ce / len(loader)
    avg_loss_supcon = total_loss_supcon / len(loader)
    accuracy = correct / total
    return avg_loss, avg_loss_ce, avg_loss_supcon, accuracy, np.concatenate(all_preds), np.concatenate(all_targets)


# ---------- 详细测试评估（Top-1、Top-5、F1）----------
@torch.no_grad()
def detailed_test_evaluate(model, loader, loss_fn_ce, loss_fn_supcon, alpha, device):
    model.eval()
    total_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='Test  ', leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs, emb = model(imgs)
        loss_ce = loss_fn_ce(outputs, targets)
        loss_supcon = loss_fn_supcon(emb, targets)
        loss = loss_ce + alpha*loss_supcon

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


# ---------- 主流程 ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    print(f'[Config] device={args.device}  batch_size={args.batch_size}  '
          f'epochs={args.epochs}  lr={args.lr}  warmup_epochs={args.warmup_epochs}  '
          f'cutmix_prob={args.cutmix_prob}  cutmix_alpha={args.cutmix_alpha}  save_dir={args.save_dir}')

    os.makedirs(args.save_dir, exist_ok=True)

    save_config(args)

    full_train = torchvision.datasets.CIFAR100(args.data_dir, train=True, download=True)
    # 统一划分索引
    train_size = len(full_train) - 10000
    val_size = 10000
    train_idx, val_idx = random_split(
        range(len(full_train)),
        [train_size, val_size],
        generator=generator
    )

    # 分别构造带增强训练集、无增强验证集
    train_data = torchvision.datasets.CIFAR100(
        args.data_dir, train=True, transform=CIFAR100_Transform(split='train'))
    train_data = torch.utils.data.Subset(train_data, train_idx)

    val_data = torchvision.datasets.CIFAR100(
        args.data_dir, train=True, transform=CIFAR100_Transform(split='val'))
    val_data = torch.utils.data.Subset(val_data, val_idx)
    
    test_data = torchvision.datasets.CIFAR100(args.data_dir, train=False,
                                             transform=CIFAR100_Transform(split='test'), download=True)

    print(f'Train samples: {len(train_data)}  |  Val samples: {len(val_data)}  |  Test samples: {len(test_data)}')

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    # 模型、损失函数、优化器
    model = Qiming_4_5().to(args.device)
    loss_fn_ce = CrossEntropyLoss(label_smoothing=0.1).to(args.device) # 采用标签平滑，增加负样本的惩罚项
    loss_fn_supcon = SupConLoss(temperature=0.07)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=8e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 学习率调度器
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     patience=3,
    #     factor=0.5
    #     )

    # 学习率预热 + 余弦退火
    # 预热阶段（LinearLR）：前 warmup_epochs 个 epoch 从极小学习率线性升到 --lr
    # 余弦阶段（CosineAnnealingLR）：预热结束后退火到 eta_min
    # 用 SequentialLR 标准库将两者串联，自动按 epoch 切换
    warmup_epochs = max(0, args.warmup_epochs)

    # 余弦退火的 T_max 设为「预热之后的剩余轮次」，避免总周期超出训练长度
    cosine_T_max = max(1, args.epochs - warmup_epochs)
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cosine_T_max,
        eta_min=1e-5
    )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1e-3,   # 预热起始 lr = lr * 1e-3，避免初期步长过大
            end_factor=1.0,      # 预热结束 lr = --lr
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_annealing],
            milestones=[warmup_epochs]   # 第 warmup_epochs 个 epoch 后切换到余弦退火
        )
        print(f'[Scheduler] Warmup {warmup_epochs} epochs (LinearLR) -> CosineAnnealingLR (T_max={cosine_T_max})')
    else:
        scheduler = cosine_annealing
        print(f'[Scheduler] CosineAnnealingLR (T_max={cosine_T_max}), warmup disabled')

    # 训练循环
    train_losses, train_losses_ce, train_losses_supcon = [], [], []
    val_losses, val_losses_ce, val_losses_supcon = [], [], []
    lr_history = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        tr_loss, tr_loss_ce, tr_loss_supcon = train_epoch(
            model,
            train_loader,
            loss_fn_ce,
            loss_fn_supcon,
            args.alpha,
            optimizer,
            args.device,
            epoch,
            args.epochs,
            cutmix_prob=args.cutmix_prob,
            cutmix_alpha=args.cutmix_alpha
        )
        val_loss, val_loss_ce, val_loss_supcon, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            loss_fn_ce,
            loss_fn_supcon,
            args.alpha,
            args.device, epoch,
            args.epochs,
            )

        train_losses.append(tr_loss)
        train_losses_ce.append(tr_loss_ce)
        train_losses_supcon.append(tr_loss_supcon)
        val_losses.append(val_loss)
        val_losses_ce.append(val_loss_ce)
        val_losses_supcon.append(val_loss_supcon)
        lr_history.append(scheduler.get_last_lr()[0])

        print(f'>>> Epoch {epoch + 1:2d}/{args.epochs}  |  '
              f'Train Loss: {tr_loss:.4f} (CE:{tr_loss_ce:.4f}, SupCon:{tr_loss_supcon:.4f})  |  '
              f'Val Loss: {val_loss:.4f} (CE:{val_loss_ce:.4f}, SupCon:{val_loss_supcon:.4f})  |  '
              f'Val Acc: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'    [*] Best model saved (val_loss={val_loss:.4f})')

        scheduler.step()

    # 保存训练曲线（三个损失 + 学习率）
    plot_training_curves(
        train_losses, train_losses_ce, train_losses_supcon,
        val_losses, val_losses_ce, val_losses_supcon,
        lr_history,
        os.path.join(args.save_dir, 'training_curves.png')
    )

    # 加载验证集最优模型权重再进行测试
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    print(f'Loaded best model (val_loss={best_val_loss:.4f}) for test evaluation.')

    # 训练完成后，在测试集上做最终评估
    print('Evaluating on test set...')
    test_loss, top1_acc, top5_acc, macro_f1, test_preds, test_targets = \
        detailed_test_evaluate(
            model, 
            test_loader, 
            loss_fn_ce,
            loss_fn_supcon,
            args.alpha, 
            args.device,
            )

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
