# ============ data/dataset.py ============
"""数据集加载：ImageFolder → DataLoader，含类别权重计算"""
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CONFIG
from data.augmentations import get_train_transforms, get_val_transforms


def prepare_data():
    """
    将 datasets/ 只读数据复制到 .data/weather/ 可写目录（仅首次运行）

    Returns:
        str: 可写数据目录路径
    """
    src = CONFIG["data_root"]
    dst = CONFIG["writable_root"]

    if not os.path.exists(dst):
        print(f"Copying dataset from {src} to {dst} ...")
        shutil.copytree(src, dst)
        print("Dataset copy complete.")
    else:
        print(f"Dataset already exists at {dst}")

    return dst


def create_dataloaders(data_root=None, img_size=None, batch_size=None, num_workers=None):
    """
    创建训练和验证 DataLoader

    Args:
        data_root: str — 数据根目录（默认从 CONFIG 读取）
        img_size: int — 图片尺寸
        batch_size: int — 批次大小
        num_workers: int — 数据加载线程数

    Returns:
        tuple: (train_loader, val_loader, class_counts)
    """
    cfg = CONFIG
    root = data_root or prepare_data()
    size = img_size or cfg["img_size"]
    bs = batch_size or cfg["batch_size"]
    nw = num_workers or cfg["num_workers"]

    # 全量加载以获取类别分布
    full_dataset = ImageFolder(root)
    num_classes = len(full_dataset.classes)
    class_counts = np.bincount(full_dataset.targets, minlength=num_classes)

    # 分层划分训练集/验证集
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg["val_split"],
        stratify=full_dataset.targets,
        random_state=cfg["seed"],
    )

    # 分别创建两个 ImageFolder 实例（不同 transform）+ Subset
    train_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_train_transforms(size)),
        train_idx,
    )
    val_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_val_transforms(size)),
        val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
    )

    print(f"Classes: {full_dataset.classes}")
    print(f"Class distribution: {dict(zip(full_dataset.classes, class_counts.astype(int)))}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_loader, val_loader, class_counts


def compute_class_weights(class_counts):
    """
    根据类别样本数计算 FocalLoss 的 alpha 参数

    Args:
        class_counts: np.ndarray — 各类别样本数

    Returns:
        torch.Tensor: 归一化的 alpha 权重
    """
    alpha = 1.0 / (class_counts + 1e-8)
    alpha = alpha / alpha.sum() * len(class_counts)
    return torch.tensor(alpha, dtype=torch.float32)
