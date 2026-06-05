# ============ data/augmentations.py ============
"""train/val 数据增强策略"""
import torchvision.transforms as transforms


def get_train_transforms(img_size=456):
    """
    训练集增强：保守几何裁剪 + 水平翻转 + 轻量 RandAugment + 归一化

    - RandomResizedCrop scale=(0.85, 1.0)：保留至少 85% 原图，避免裁剪掉雾/云/雨痕
    - RandAugment magnitude=3：大幅降低颜色操作强度，防止 Solarize/Equalize/Posterize
      破坏天气判别特征（雾被增亮消失、云被色调扭曲、雨痕被锐化模糊）

    Args:
        img_size: int — 输入尺寸（默认 456，EfficientNet-B5 原生分辨率）

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(img_size=456):
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
