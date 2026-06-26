import os
import torch
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dir(path):
    if not os.path.exists(path):
        print('路径', path, '不存在，创建新目录')
        os.makedirs(path, exist_ok=True)

def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_encoder(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in ckpt.items():
        if 'backbone' in k:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    return model

def calculate_acc(logits, labels):
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()
    return acc

# 绘制训练过程损失曲线
def plot_train_curve(train_loss, root, epoch):

    plt.figure(figsize=(9,6), dpi=150)

    x = list(range(1, len(train_loss)+1))

    plt.plot(x,train_loss, marker='o', markersize=3)
    plt.title('Training Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)

    save_root = os.path.join(root, f'loss-curve-epoch{epoch}.png')

    plt.savefig(save_root, bbox_inches='tight')
    print(f'损失曲线已保存至{save_root}')

# 加载模型权重函数
def load_checkpoint(model, ckpt_path):
    """加载训练好的权重"""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'成功加载权重: {ckpt_path}')
    return model

# 保存超参数配置
def save_config(args):

    os.makedirs(args.save_dir, exist_ok=True)

    # 保存超参数配置到 YAML
    config = {
        'base_dir': args.base_dir,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'lr_rate': args.lr_rate,
        'temperature': args.temperature,
        'seed': args.seed,
    }
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f'[INFO] Config saved to {os.path.join(args.save_dir, "config.yaml")}')