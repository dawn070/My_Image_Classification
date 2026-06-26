import os
import argparse
import torch
from model import SimCLR
from utils import set_seed, create_dir, save_checkpoint, plot_train_curve, save_config
from dataset import get_train_dataloders
from loss import NTXenLoss

from tqdm import tqdm

# 训练的主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./SimCLR', help='基础目录')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='数据集所在文件夹')
    parser.add_argument('--save_dir', type=str, default='./SimCLR/runs/temp')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='每轮批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率初值')
    parser.add_argument('--lr_rate', type=float, default=0.01, help='余弦退火的最终学习率倍率')
    parser.add_argument('--temperature', type=float, default=0.5, help='NTXenLoss的温度缩放系数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help="Dataloder进程数")
    args = parser.parse_args()

    save_config(args=args)

    set_seed(args.seed)
    base_dir = args.base_dir
    save_dir = args.save_dir
    create_dir(save_dir)
    ckpt_path = os.path.join(save_dir, 'simclr_checkpoints')
    create_dir(ckpt_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    bs = args.batch_size
    end_lr = args.lr*args.lr_rate
    data_path = args.data_dir
    train_loader = get_train_dataloders(batch_size=bs, root=data_path, num_workers=args.num_workers)
    # print('train_loader长度：', len(train_loader))

    # 加载模型
    model = SimCLR(out_dim=128).to(device)
    # 损失函数
    loss_fn = NTXenLoss(temp=args.temperature)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min=end_lr
    )

    loss_history = []

    # 训练循环
    print('开始训练······')
    for epoch in range(args.epochs):

        pbar = tqdm(train_loader, f'Epoch {epoch+1}/{args.epochs} [Train]', leave=False)

        model.train()
        total_loss = 0.0

        for v1, v2 in pbar:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)  # 返回经过投影头的 z
            _, z2 = model(v2)

            loss = loss_fn(z1, z2)

            optimizer.zero_grad()  # 每次循环梯度清零
            loss.backward()  # 计算梯度，为更新做准备
            optimizer.step()  # 更新参数

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'>>> Epoch{epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}')

        loss_history.append(avg_loss)
        scheduler.step()

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, os.path.join(ckpt_path, f'checkpoint{epoch+1}.pth'))
            plot_train_curve(loss_history, save_dir, epoch)            

    save_checkpoint(model, os.path.join(ckpt_path, f'last_checkpoint{epoch+1}.pth'))
    plot_train_curve(loss_history, save_dir, args.epochs)



if __name__ == "__main__":
    main()