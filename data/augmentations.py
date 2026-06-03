# ============ data/augmentations.py ============
"""train/val 数据增强策略"""
import torchvision.transforms as transforms


def get_train_transforms(img_size=224):
    """
    训练集增强：RandomResizedCrop + 水平翻转 + RandAugment + 归一化

    Args:
        img_size: int — 输入尺寸

    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(img_size=224):
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
