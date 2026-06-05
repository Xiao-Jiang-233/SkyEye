# ============ data/augmentations.py ============
"""train/val 数据增强策略"""
import torchvision.transforms as transforms


def get_train_transforms(img_size=380):
    """
    训练集增强：保守几何裁剪 + 水平翻转 + 轻量 RandAugment + 归一化

    - RandomResizedCrop scale=(0.85, 1.0)：保留至少 85% 原图，避免裁剪掉雾/云/雨痕
    - RandAugment magnitude=9：对标 EfficientNet 训练配方（Cubuk et al., NeurIPS 2020）
      B4@380 介于 B3(300, M=9-10) 和 B5(456, M=15) 之间，M=9 为保守下限

    Args:
        img_size: int — 输入尺寸（默认 380，EfficientNet-B4 原生分辨率）

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(img_size=380):
    """
    验证集增强：Resize + CenterCrop + 归一化

    Args:
        img_size: int — 输入尺寸

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
