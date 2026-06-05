# ============ training/train_teacher.py ============
"""
教师模型训练脚本

优化策略：
  ① cloudy 过采样 2×（data/dataset.py）
  ② FocalLoss γ 降为 1（config.py）—— 让困难样本拿到梯度
  ③ SAM 优化器 rho=0.05（平坦极小值 → 泛化好）
  ④ EMA 权重指数滑动平均 decay=0.99997（几乎免费）
  ⑤ BF16 autocast + 梯度裁剪（无需 GradScaler，RTX 5070 原生支持）

输出：results/teacher_best.pth
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.dataset import create_dataloaders, compute_class_weights
from utils.logger import TrainLogger


# ============================================================
# MixUp 数据增强 (Zhang et al., ICLR 2018)
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """
    MixUp: x̃ = λ·xᵢ + (1-λ)·xⱼ, ỹ = λ·yᵢ + (1-λ)·yⱼ

    λ ~ Beta(α, α)，α 控制混合强度：
      - α → 0: 趋近原始样本（弱正则化）
      - α = 0.2: ImageNet 标准值
      - α = 1.0: Uniform(0,1)，最强正则化

    Args:
        x: Tensor (B, C, H, W) — 输入图像批次
        y: Tensor (B,) — 标签（类索引）
        alpha: float — Beta 分布参数，0.0 关闭 MixUp

    Returns:
        mixed_x: Tensor — 混合后的图像
        y_a, y_b: Tensor — 两个原始标签
        lam: float — λ 权重
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class FocalLoss(nn.Module):
    """
    Focal Loss for 类别不平衡 + label smoothing

    FL = -(1 - pt)^γ × CE_smoothed

    其中 CE_smoothed 由 F.cross_entropy(label_smoothing=ε) 提供，
    pt 仍从模型对真实类的 softmax 概率计算，保持 focal 调制语义。

    Args:
        alpha: Tensor — 各类别权重
        gamma: float — 聚焦参数（默认 1.0）
        label_smoothing: float — 标签平滑 ε（默认 0.0 即关闭）
        reduction: str — 'mean' / 'sum'
    """

    def __init__(self, alpha=None, gamma=1.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # label_smoothing 直接交给 cross_entropy 处理
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing,
        )
        # pt: 模型对真实类的预测概率（保持 focal 调制语义）
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ============================================================
# SAM (Sharpness-Aware Minimization)
# ============================================================
class SAM:
    """
    SAM 优化器包装器：寻找平坦极小值，提升泛化性能

    原理：先在梯度方向做一步扰动（w + ε·g/||g||），在该点计算损失
          再回退并对扰动点的梯度做 optimizer.step()
          训练时间 ×2（两次 forward + backward），但 70 分钟内仍可接受

    Args:
        base_optimizer: torch.optim.Optimizer — 基础优化器（如 AdamW）
        rho: float — 扰动半径，控制平坦程度（默认 0.05）

    Usage (without GradScaler — SAM + autocast 社区标准做法):
        # First pass
        loss1 = model(x)
        loss1.backward()
        sam.first_step()
        optimizer.zero_grad()

        # Second pass
        loss2 = model(x)
        loss2.backward()
        sam.second_step()
    """

    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self._eps_cache = {}  # id(param) → perturbation tensor

    @torch.no_grad()
    def first_step(self):
        """梯度上升扰动：w ← w + ρ·g/||g||"""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norm = grad.norm(p=2)
                if grad_norm > 1e-12:
                    eps = self.rho * grad / grad_norm
                    p.add_(eps)
                    self._eps_cache[id(p)] = eps

    @torch.no_grad()
    def second_step(self):
        """撤销扰动 + 优化器步进"""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                eps = self._eps_cache.pop(id(p), None)
                if eps is not None:
                    p.sub_(eps)
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
class EMA:
    """
    模型权重指数滑动平均

    shadow = decay × shadow + (1 - decay) × current

    Args:
        model: nn.Module
        decay: float — 衰减率（默认 0.99997，平滑窗口 ~33k steps ≈ 7 epochs）

    Usage:
        ema = EMA(model, decay=0.999)
        # 每步训练后
        ema.update(model)
        # 验证前
        ema.apply_shadow(model)
        # 验证后
        ema.restore(model)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """保存当前训练权重，替换为 EMA 平滑权重（用于验证）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """恢复训练权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._backup[name])


# ============================================================
# Evaluate
# ============================================================
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


# ============================================================
# Train Teacher
# ============================================================
def train_teacher():
    """训练教师模型主函数"""
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 1) 数据加载（DRW: 两个训练集，前 60% epoch 不用过采样，后 40% 开启）
    #    DRW = Deferred Re-weighting/Re-sampling（LDAM, NeurIPS 2019）
    #    过采样从 epoch 0 开始会导致 cloudy 过度自信（高召回低精确），
    #    延迟到后期再开启，让模型先学好特征表示，再校准决策边界
    train_loader_std, val_loader, class_counts, class_names = create_dataloaders(cloudy_oversample=False)
    train_loader_os, _, _, _ = create_dataloaders(cloudy_oversample=True)

    drw_start_epoch = int(cfg["teacher_epochs"] * 0.6)  # 后 40% epoch 开启过采样

    # 2) 创建模型
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["teacher_pretrained"],
    ).to(device)

    # 3) 损失函数（FocalLoss + 类别权重 — 始终用原始分布计算 alpha）
    #    DRW 只改变采样分布，不改 loss 权重，避免叠加导致过矫正
    alpha = compute_class_weights(class_counts)
    criterion = FocalLoss(
        alpha=alpha, gamma=cfg["focal_gamma"],
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    # 4) 优化器 + SAM 包装 + 调度器
    base_optimizer = optim.AdamW(
        teacher.parameters(),
        lr=cfg["teacher_lr"],
        weight_decay=cfg["teacher_weight_decay"],
    )
    sam = SAM(base_optimizer, rho=cfg.get("sam_rho", 0.05))  # ④ SAM
    # 调度器：Linear warmup → CosineAnnealing
    warmup_epochs = cfg.get("warmup_epochs", 2)
    if warmup_epochs > 0:
        warmup = LinearLR(base_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(base_optimizer, T_max=cfg["teacher_epochs"] - warmup_epochs)
        scheduler = SequentialLR(base_optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(base_optimizer, T_max=cfg["teacher_epochs"])

    # ⑤ EMA 权重平滑
    ema = EMA(teacher, decay=cfg.get("ema_decay", 0.999))

    # TensorBoard 日志
    logger = TrainLogger(log_dir="results/tb_results/teacher", use_tb=cfg["use_tb"])

    # 5) 训练循环
    best_f1 = 0.0
    # BF16: 与 FP32 相同动态范围，无需 GradScaler，RTX 5070+ 原生支持
    use_amp = cfg["fp16"] and torch.cuda.is_available()

    sam_start_epoch = cfg["teacher_epochs"] - 5  # 后 5 轮启用 SAM，给足 Phase I+II 收敛时间
    overfit_warn_threshold = 0.05  # train_f1 - val_f1 > 5% 时告警
    mixup_alpha = cfg.get("mixup_alpha", 0.0)  # MixUp 混合强度 (Zhang et al., ICLR 2018)

    print(f"\nDRW schedule: epoch 0-{drw_start_epoch-1} standard, epoch {drw_start_epoch}-{cfg['teacher_epochs']-1} cloudy oversample")
    print(f"SAM schedule: epoch 0-{sam_start_epoch-1} Fast, epoch {sam_start_epoch}-{cfg['teacher_epochs']-1} SAM")
    print(f"MixUp: {'alpha=' + str(mixup_alpha) if mixup_alpha > 0 else 'OFF'}\n")

    for epoch in range(cfg["teacher_epochs"]):
        use_sam = epoch >= sam_start_epoch
        use_os = epoch >= drw_start_epoch  # DRW 延迟过采样
        mode_parts = []
        if use_sam:
            mode_parts.append("SAM")
        else:
            mode_parts.append("Fast")
        if use_os:
            mode_parts.append("OS")
        if mixup_alpha > 0:
            mode_parts.append("MU")
        mode_tag = "+".join(mode_parts)

        # DRW: 根据阶段选择 DataLoader
        train_loader = train_loader_os if use_os else train_loader_std

        # --- Train ---
        teacher.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []  # 收集训练集预测用于过拟合监控
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{cfg['teacher_epochs']} [{mode_tag}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # ⑥ MixUp: 生成虚拟混合样本 (Zhang et al., ICLR 2018)
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

            if use_sam:
                # ---- SAM first pass: MixUp forward-backward ----
                sam.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = teacher(mixed_images)
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                sam.first_step()
                base_optimizer.zero_grad()

                # ---- SAM second pass: perturbed MixUp forward-backward ----
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits_perturbed = teacher(mixed_images)
                    loss_perturbed = lam * criterion(logits_perturbed, labels_a) + (1 - lam) * criterion(logits_perturbed, labels_b)
                loss_perturbed.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                sam.second_step()

                # 用 first pass（未扰动）的预测做监控（labels_a 为主标签）
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels_list.extend(labels_a.cpu().numpy())
            else:
                # ---- Fast mode: MixUp forward-backward ----
                base_optimizer.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = teacher(mixed_images)
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                base_optimizer.step()

                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels_list.extend(labels_a.cpu().numpy())

            # ⑤ EMA 更新（每步）
            ema.update(teacher)

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # 计算训练集 F1（过拟合监控）
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        # --- Validate（使用 EMA 权重）---
        ema.apply_shadow(teacher)
        val_f1, val_acc, per_class_f1 = evaluate(
            teacher, val_loader, device, class_names,
        )
        ema.restore(teacher)

        avg_loss = train_loss / len(train_loader)
        gap = train_f1 - val_f1
        gap_str = f"│ Gap={gap:+.4f}"
        if gap > overfit_warn_threshold:
            gap_str += f" ⚠️ OVERFIT"
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} "
              f"│ Train F1={train_f1:.4f} │ Val F1={val_f1:.4f} {gap_str} │ Val Acc={val_acc:.2f}%")

        # TensorBoard 记录（F1 为主监控，含 per-class + gap）
        logger.log_metrics("train", {"loss": avg_loss, "F1_Macro": train_f1}, epoch + 1)
        val_metrics = {"F1_Macro": val_f1, "Acc": val_acc, "Overfit_Gap": gap}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, epoch + 1)

        # 保存最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 保存 EMA 平滑后的权重
            ema.apply_shadow(teacher)
            torch.save(teacher.state_dict(), cfg["teacher_ckpt"])
            ema.restore(teacher)
            print(f"  ✓ Best teacher saved (EMA)! F1={best_f1:.4f}")

    print(f"\nTeacher training done. Best F1: {best_f1:.4f}")

    logger.close()

    # 加载最佳 EMA 权重
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], weights_only=False))
    return teacher


if __name__ == "__main__":
    train_teacher()
