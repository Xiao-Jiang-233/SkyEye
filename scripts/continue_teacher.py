# ============ scripts/continue_teacher.py ============
"""
从 checkpoint 继续训练教师模型，支持 Fast 和 SAM 两种模式

用法：
    # Fast+OS+MU（无 SAM）
    python scripts/continue_teacher.py results/checkpoints/teacher/teacher_epoch_14.pth --epochs 5

    # SAM+OS（从 epoch 14 继续，SAM 5 轮收尾，关闭 MixUp）
    python scripts/continue_teacher.py results/checkpoints/teacher/teacher_epoch_14.pth --epochs 5 --use_sam
"""
import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.dataset import create_dataloaders, compute_class_weights
from utils.logger import TrainLogger

# 复用 train_teacher 中的组件
from training.train_teacher import FocalLoss, EMA, SAM, mixup_data, evaluate


def continue_teacher(checkpoint_path, extra_epochs=5, lr=None, mixup_alpha=0.2, use_sam=False,
                     sam_rho=0.05, sam_mixup_alpha=0.0):  # SAM 阶段关闭 MixUp
    """从 checkpoint 继续训练教师模型"""
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Extra epochs: {extra_epochs}")
    print(f"Mode: {'SAM+OS' if use_sam else 'Fast+OS+MU'}")

    # 1) 数据加载 — 使用 oversample 版本（DRW 阶段）
    print("\n--- Loading Data ---")
    _, val_loader, class_counts, class_names = create_dataloaders(cloudy_oversample=False)
    train_loader_os, _, _, _ = create_dataloaders(cloudy_oversample=True)
    print(f"Cloudy oversampling: ON (2x)")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader_os.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 2) 创建模型并加载 checkpoint
    print("\n--- Loading Model ---")
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,  # 从 checkpoint 加载，无需 pretrained
    ).to(device)

    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    teacher.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    # 3) 损失函数
    alpha = compute_class_weights(class_counts)
    criterion = FocalLoss(alpha=alpha, gamma=cfg["focal_gamma"])

    # 4) 优化器 — 从当前 checkpoint 重新初始化
    cont_lr = lr if lr is not None else (2e-5 if use_sam else 5e-5)  # SAM 阶段用更低的 LR
    base_optimizer = optim.AdamW(
        teacher.parameters(), lr=cont_lr, weight_decay=cfg["teacher_weight_decay"]
    )

    # SAM 包装器
    sam = SAM(base_optimizer, rho=sam_rho) if use_sam else None

    scheduler = CosineAnnealingLR(base_optimizer, T_max=extra_epochs)

    # 5) EMA — 从当前权重初始化 shadow
    ema = EMA(teacher, decay=cfg.get("ema_decay", 0.999))

    # TensorBoard
    logger = TrainLogger(log_dir="results/tb_results/teacher", use_tb=cfg["use_tb"])

    # 训练配置
    use_amp = cfg["fp16"] and torch.cuda.is_available()
    best_f1 = 0.0
    backup_dir = cfg["teacher_ckpt_dir"]

    # SAM 阶段用轻量 MixUp
    epoch_mixup = sam_mixup_alpha if use_sam else mixup_alpha

    print(f"\nLR: {cont_lr}, MixUp alpha: {epoch_mixup}, SAM rho: {sam_rho if use_sam else 'N/A'}")
    print(f"BF16 AMP: {'ON' if use_amp else 'OFF'}\n")

    for epoch in range(extra_epochs):
        mode_tag = "SAM+OS" if use_sam else "Fast+OS+MU"

        teacher.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        pbar = tqdm(train_loader_os, desc=f"Continue Epoch {epoch+1}/{extra_epochs} [{mode_tag}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # MixUp
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, epoch_mixup)

            if use_sam:
                # ---- SAM first pass ----
                sam.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = teacher(mixed_images)
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                sam.first_step()
                base_optimizer.zero_grad()

                # ---- SAM second pass ----
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits_perturbed = teacher(mixed_images)
                    loss_perturbed = lam * criterion(logits_perturbed, labels_a) + (1 - lam) * criterion(logits_perturbed, labels_b)
                loss_perturbed.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                sam.second_step()

                # 用 first pass 的预测做监控
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels_list.extend(labels_a.cpu().numpy())
                train_loss += loss.item()
            else:
                # ---- Fast mode ----
                base_optimizer.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = teacher(mixed_images)
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                base_optimizer.step()

                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels_list.extend(labels_a.cpu().numpy())
                train_loss += loss.item()

            # EMA
            ema.update(teacher)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        # Validate (EMA)
        ema.apply_shadow(teacher)
        val_f1, val_acc, per_class_f1 = evaluate(teacher, val_loader, device, class_names)
        ema.restore(teacher)

        avg_loss = train_loss / len(train_loader_os)
        gap = train_f1 - val_f1
        gap_str = f"│ Gap={gap:+.4f}"
        if gap > 0.05:
            gap_str += " ⚠️ OVERFIT"

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
              f"│ Train F1={train_f1:.4f} │ Val F1={val_f1:.4f} {gap_str} │ Val Acc={val_acc:.2f}%")

        # TensorBoard
        logger.log_metrics("train", {"loss": avg_loss, "F1_Macro": train_f1}, epoch)
        val_metrics = {"F1_Macro": val_f1, "Acc": val_acc, "Overfit_Gap": gap}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, epoch)

        # 保存最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            ema.apply_shadow(teacher)
            torch.save(teacher.state_dict(), cfg["teacher_ckpt"])
            ema.restore(teacher)
            print(f"  ✓ Best teacher saved (EMA)! F1={best_f1:.4f}")

        # 周期备份
        ckpt_path = f"{backup_dir}/teacher_cont_{epoch:02d}.pth"
        ema.apply_shadow(teacher)
        torch.save(teacher.state_dict(), ckpt_path)
        ema.restore(teacher)

    print(f"\nContinue training done. Best F1: {best_f1:.4f}")
    logger.close()

    # 加载最佳
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], weights_only=True))
    return teacher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue teacher training from checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pth file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of extra epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-5 for SAM, 5e-5 for Fast)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="MixUp alpha (Fast mode)")
    parser.add_argument("--use_sam", action="store_true", help="Enable SAM optimizer")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM perturbation radius")
    parser.add_argument("--sam_mixup_alpha", type=float, default=0.05,
                        help="MixUp alpha during SAM phase (light, to avoid over-regularization)")
    args = parser.parse_args()

    continue_teacher(
        args.checkpoint, args.epochs, args.lr, args.mixup_alpha,
        args.use_sam, args.sam_rho, args.sam_mixup_alpha,
    )
