# ============ training/train_teacher.py ============
"""
教师模型训练脚本

使用 FocalLoss 处理类别不平衡 + 混合精度训练 + Cosine 调度
输出：results/teacher_best.pth
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.dataset import create_dataloaders, compute_class_weights
from utils.logger import TrainLogger


class FocalLoss(nn.Module):
    """
    Focal Loss for 类别不平衡

    FL = -(1 - pt)^γ × log(pt)

    Args:
        alpha: Tensor — 各类别权重
        gamma: float — 聚焦参数（默认 2.0）
        reduction: str — 'mean' / 'sum'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    """
    在验证集上评估模型

    Returns:
        tuple: (macro_f1, accuracy, per_class_f1_dict | None)
    """
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100

    if class_names is not None:
        per_class_f1 = dict(zip(class_names, per_class_f1))
    return f1, acc, per_class_f1


def train_teacher():
    """训练教师模型主函数"""
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 1) 数据加载
    train_loader, val_loader, class_counts = create_dataloaders()

    # 2) 创建模型
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["teacher_pretrained"],
    ).to(device)

    # 3) 损失函数（FocalLoss + 类别权重）
    alpha = compute_class_weights(class_counts)
    criterion = FocalLoss(alpha=alpha, gamma=cfg["focal_gamma"])

    # 4) 优化器 + 调度器
    optimizer = optim.AdamW(
        teacher.parameters(),
        lr=cfg["teacher_lr"],
        weight_decay=cfg["teacher_weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["teacher_epochs"])
    scaler = GradScaler(enabled=cfg["fp16"])

    # TensorBoard 日志
    logger = TrainLogger(log_dir="results/tb_results/teacher", use_tb=cfg["use_tb"])

    # 5) 训练循环
    best_f1 = 0.0
    for epoch in range(cfg["teacher_epochs"]):
        # --- Train ---
        teacher.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{cfg['teacher_epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg["fp16"]):
                logits = teacher(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validate ---
        val_f1, val_acc, per_class_f1 = evaluate(teacher, val_loader, device, train_loader.dataset.dataset.classes)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Val F1={val_f1:.4f} | Val Acc={val_acc:.2f}%")

        # TensorBoard 记录（F1 为主监控，含 per-class）
        logger.log_metrics("train", {"loss": avg_loss}, epoch + 1)
        val_metrics = {"F1_Macro": val_f1, "Acc": val_acc}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, epoch + 1)

        # Mo 平台 JSON 指标（Job 训练时自动可视化）
        print('{"metric": "teacher_train_loss", "value": %.4f, "epoch": %d}' % (avg_loss, epoch + 1))
        print('{"metric": "teacher_val_f1", "value": %.4f, "epoch": %d}' % (val_f1, epoch + 1))
        print('{"metric": "teacher_val_acc", "value": %.2f, "epoch": %d}' % (val_acc, epoch + 1))

        # 保存最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(teacher.state_dict(), cfg["teacher_ckpt"])
            print(f"  ✓ Best teacher saved! F1={best_f1:.4f}")

    print(f"\nTeacher training done. Best F1: {best_f1:.4f}")

    logger.close()

    # 加载最佳权重
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], weights_only=False))
    return teacher


if __name__ == "__main__":
    train_teacher()
