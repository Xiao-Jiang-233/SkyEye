# ============ training/train_teacher.py ============
"""
教师模型训练脚本

优化策略：
  ① cloudy 过采样 2×（data/dataset.py）
  ② FocalLoss γ=1.0 —— 让困难样本拿到梯度
  ③ EMA 权重指数滑动平均 decay=0.99997（几乎免费）
  ④ BF16 autocast + 梯度裁剪（无需 GradScaler，RTX 5070 原生支持）

输出：results/teacher_best.pth
"""
import os
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

    支持 per-class label smoothing（方案 D）：dict 格式按类名分配不同平滑值。

    Args:
        alpha: Tensor — 各类别权重
        gamma: float — 聚焦参数（默认 1.0）
        label_smoothing: float | dict — 标签平滑 ε。float 全局统一；dict 按类名映射
        reduction: str — 'mean' / 'sum'
        class_names: list — 类名列表（per-class smoothing 时需要）
    """

    def __init__(self, alpha=None, gamma=1.0, label_smoothing=0.0, reduction='mean', class_names=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_names = class_names or []

        if isinstance(label_smoothing, dict):
            self.per_class_ls = torch.zeros(len(class_names) if class_names else 6)
            for cls_name, eps in label_smoothing.items():
                if cls_name in self.class_names:
                    self.per_class_ls[self.class_names.index(cls_name)] = eps
            self.label_smoothing = 0.0
        else:
            self.per_class_ls = None
            self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.per_class_ls is not None and self.class_names:
            smoothing_vals = self.per_class_ls.to(inputs.device)[targets]
            n_classes = inputs.size(1)
            log_probs = torch.log_softmax(inputs, dim=1)
            ce_onehot = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce_uniform = -log_probs.mean(dim=1)
            ce_loss = (1 - smoothing_vals) * ce_onehot + smoothing_vals * ce_uniform
        else:
            ce_loss = nn.functional.cross_entropy(
                inputs, targets, reduction='none', label_smoothing=self.label_smoothing,
            )

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
# ConfusionPenaltyLoss（方案 B）— 对特定混淆方向施加额外惩罚
# ============================================================
class ConfusionPenaltyLoss(nn.Module):
    """
    包装现有 loss，叠加混淆矩阵惩罚项

    L = base_loss + λ × Σ p(y'≠y) × M[y][y']

    Args:
        base_criterion: nn.Module — 基础损失函数
        penalty_matrix: Tensor (C, C) — 混淆惩罚权重矩阵
        penalty_weight: float — 惩罚项强度 λ
        class_names: list
    """

    def __init__(self, base_criterion, penalty_matrix, penalty_weight=0.3, class_names=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.penalty_weight = penalty_weight
        self.class_names = class_names or []
        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, inputs, targets):
        base_loss = self.base_criterion(inputs, targets)

        if self.penalty_weight > 0 and self.penalty_matrix is not None:
            probs = torch.softmax(inputs, dim=1)
            penalty = (probs * self.penalty_matrix[targets]).sum(dim=1)
            return base_loss + self.penalty_weight * penalty.mean()
        return base_loss


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
def _build_logit_bias(device, class_names, cfg=None):
    """从 config 构建 per-class logit bias 张量（方案 A）"""
    bias = torch.zeros(len(class_names), device=device)
    bias_cfg = (cfg or CONFIG).get("logit_bias", {})
    if bias_cfg:
        for cls_name, val in bias_cfg.items():
            if cls_name in class_names:
                bias[class_names.index(cls_name)] = val
    return bias


@torch.no_grad()
def evaluate(model, loader, device, class_names=None):
    """
    在验证集上评估模型

    支持 logit adjustment（方案 A）：推理时对 logit 做先验偏移

    Returns:
        tuple: (macro_f1, accuracy, per_class_f1_dict | None)
    """
    model.eval()
    all_preds, all_labels = [], []
    logit_bias = _build_logit_bias(device, class_names) if class_names else None
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if logit_bias is not None:
            logits = logits + logit_bias.unsqueeze(0)
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
# Phase helpers
# ============================================================
def _get_phase_paths(cfg):
    """返回 (ckpt_dir, phase1_best, phase2_best)"""
    ckpt_dir = cfg["teacher_ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    p1 = os.path.join(ckpt_dir, "fast_mu_best.pth")
    p2 = os.path.join(ckpt_dir, "fast_os_mu_best.pth")
    return ckpt_dir, p1, p2


def _save_epoch_ckpt(model, ema, ckpt_dir, global_epoch):
    """保存 per-epoch EMA checkpoint（自动清理 20 轮前的旧文件）"""
    path = os.path.join(ckpt_dir, f"teacher_epoch_{global_epoch:02d}.pth")
    ema.apply_shadow(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)
    old = os.path.join(ckpt_dir, f"teacher_epoch_{global_epoch-20:02d}.pth")
    if os.path.exists(old):
        os.remove(old)


def _save_best(model, ema, path, best_f1):
    """保存最佳 EMA checkpoint"""
    ema.apply_shadow(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)
    print(f"  ✓ Best saved! F1={best_f1:.4f} → {os.path.basename(path)}")


def _create_model(cfg, device, checkpoint_path=None):
    """创建模型：从 checkpoint 加载 或 pretrained 初始化"""
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=(checkpoint_path is None and cfg["teacher_pretrained"]),
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        teacher.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        print(f"Loaded: {checkpoint_path}")
    return teacher


# ============================================================
# Phase 1: 标准采样 + MixUp + warmup
# ============================================================
def train_teacher_phase1(teacher=None):
    """
    Phase 1: 标准采样 + MixUp α=0.2 + warmup

    Args:
        teacher: 可选，已创建的模型（链式调用时传入，None 则自动创建）
    Returns:
        teacher: 加载了 phase1 best EMA 权重的模型
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])
    ckpt_dir, phase_best, _ = _get_phase_paths(cfg)
    epochs = cfg["teacher_phase1_epochs"]  # 12
    start_epoch = 0

    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"  Phase 1: Standard + MixUp — {epochs} epochs")
    print(f"{'='*60}")

    # 数据加载（标准采样）
    train_loader, val_loader, class_counts, class_names = create_dataloaders(cloudy_oversample=False)

    # 模型
    if teacher is None:
        teacher = _create_model(cfg, device, None)

    # 损失函数（方案 D: per-class label smoothing）
    alpha = compute_class_weights(class_counts)
    smoothing = cfg.get("per_class_label_smoothing", {}) or cfg.get("label_smoothing", 0.0)
    criterion = FocalLoss(
        alpha=alpha, gamma=cfg["focal_gamma"],
        label_smoothing=smoothing, class_names=class_names,
    )

    # 优化器 + warmup + cosine 调度
    optimizer = optim.AdamW(
        teacher.parameters(), lr=cfg["teacher_lr"],
        weight_decay=cfg["teacher_weight_decay"],
    )
    warmup_epochs = cfg.get("warmup_epochs", 2)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # EMA
    ema = EMA(teacher, decay=cfg.get("ema_decay", 0.999))

    # Logger
    logger = TrainLogger(log_dir="results/tb_results/teacher", use_tb=cfg["use_tb"])

    # 训练配置
    use_amp = cfg["fp16"] and torch.cuda.is_available()
    mixup_alpha = cfg.get("mixup_alpha", 0.2)
    best_f1 = 0.0

    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(class_names, class_counts.astype(int)))}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"LR: {cfg['teacher_lr']}, MixUp α: {mixup_alpha}, Warmup: {warmup_epochs} epochs\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        global_epoch = epoch  # Phase 1 从 0 开始
        mode_tag = "P1"

        teacher.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch}/{start_epoch+epochs-1} [{mode_tag}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # MixUp
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

            # Fast forward-backward
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = teacher(mixed_images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()

            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels_list.extend(labels_a.cpu().numpy())

            ema.update(teacher)
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        # Validate (EMA)
        ema.apply_shadow(teacher)
        val_f1, val_acc, per_class_f1 = evaluate(teacher, val_loader, device, class_names)
        ema.restore(teacher)

        avg_loss = train_loss / len(train_loader)
        gap = train_f1 - val_f1
        gap_str = f"│ Gap={gap:+.4f}"
        if gap > 0.05:
            gap_str += " ⚠️ OVERFIT"

        print(f"Epoch {global_epoch}: Train Loss={avg_loss:.4f} "
              f"│ Train F1={train_f1:.4f} │ Val F1={val_f1:.4f} {gap_str} │ Val Acc={val_acc:.2f}%")

        # TensorBoard
        logger.log_metrics("train", {"loss": avg_loss, "F1_Macro": train_f1}, global_epoch)
        val_metrics = {"F1_Macro": val_f1, "Acc": val_acc, "Overfit_Gap": gap}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, global_epoch)

        # 保存最佳（phase best + 全局 best）
        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_best(teacher, ema, phase_best, best_f1)
            _save_best(teacher, ema, cfg["teacher_ckpt"], best_f1)

        # 周期备份
        _save_epoch_ckpt(teacher, ema, ckpt_dir, global_epoch)

    print(f"\nPhase 1 done. Best F1: {best_f1:.4f}")
    logger.close()

    # 加载 phase best
    teacher.load_state_dict(torch.load(phase_best, weights_only=False))
    return teacher


# ============================================================
# Phase 2: DRW 过采样 + MixUp
# ============================================================
def train_teacher_phase2(teacher=None):
    """
    Phase 2: DRW 过采样 + MixUp α=0.2

    自动从 phase1 best 加载权重（如 teacher=None）。
    重新初始化优化器（低 LR）和 EMA。

    Args:
        teacher: 可选，链式调用时传入 Phase 1 返回的模型
    Returns:
        teacher: 加载了 phase2 best EMA 权重的模型
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])
    ckpt_dir, phase1_best, phase_best = _get_phase_paths(cfg)
    epochs = cfg["teacher_phase2_epochs"]  # 3
    start_epoch = cfg["teacher_phase1_epochs"]  # 12

    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"  Phase 2: DRW Oversampling + MixUp — {epochs} epochs")
    print(f"{'='*60}")

    # 数据加载（方案 C：cloudy + sunny 双过采样）
    _, val_loader, class_counts, class_names = create_dataloaders(cloudy_oversample=False)
    train_loader, _, _, _ = create_dataloaders(cloudy_oversample=True, sunny_oversample=True)

    # 模型：从 phase1 best 加载
    if teacher is None:
        teacher = _create_model(cfg, device, phase1_best)

    # 损失函数（方案 B: ConfusionPenalty + 方案 D: per-class smoothing）
    alpha = compute_class_weights(class_counts)
    smoothing = cfg.get("per_class_label_smoothing", {}) or cfg.get("label_smoothing", 0.0)
    base_criterion = FocalLoss(
        alpha=alpha, gamma=cfg["focal_gamma"],
        label_smoothing=smoothing, class_names=class_names,
    )

    # 构造混淆惩罚矩阵（sunny/rainy/foggy → cloudy）
    num_c = len(class_names)
    penalty_matrix = torch.zeros(num_c, num_c)
    if cfg.get("confusion_penalty_weight", 0) > 0:
        cloudy_idx = class_names.index("cloudy") if "cloudy" in class_names else None
        if cloudy_idx is not None:
            penalties = []
            for src_name in ["sunny", "rainy", "foggy"]:
                if src_name in class_names:
                    src_idx = class_names.index(src_name)
                    penalty_matrix[src_idx, cloudy_idx] = 1.0
                    penalties.append(f"{src_name}({src_idx})→cloudy({cloudy_idx})")
            print(f"Confusion penalty: {', '.join(penalties)} = 1.0, "
                  f"λ={cfg['confusion_penalty_weight']}")

    criterion = ConfusionPenaltyLoss(
        base_criterion, penalty_matrix,
        penalty_weight=cfg.get("confusion_penalty_weight", 0),
        class_names=class_names,
    ).to(device)

    # 优化器（半量 LR，无 warmup）
    phase2_lr = cfg["teacher_lr"] * 0.5
    optimizer = optim.AdamW(
        teacher.parameters(), lr=phase2_lr,
        weight_decay=cfg["teacher_weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA（从当前权重重新初始化 shadow）
    ema = EMA(teacher, decay=cfg.get("ema_decay", 0.999))

    # Logger
    logger = TrainLogger(log_dir="results/tb_results/teacher", use_tb=cfg["use_tb"])

    # 训练配置
    use_amp = cfg["fp16"] and torch.cuda.is_available()
    mixup_alpha = cfg.get("mixup_alpha", 0.2)
    best_f1 = 0.0

    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"LR: {phase2_lr}, MixUp α: {mixup_alpha}\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        global_epoch = epoch  # 12, 13, 14
        mode_tag = "P2"

        teacher.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        pbar = tqdm(train_loader, desc=f"P2 Epoch {epoch}/{start_epoch+epochs-1} [{mode_tag}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = teacher(mixed_images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()

            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels_list.extend(labels_a.cpu().numpy())

            ema.update(teacher)
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        ema.apply_shadow(teacher)
        val_f1, val_acc, per_class_f1 = evaluate(teacher, val_loader, device, class_names)
        ema.restore(teacher)

        avg_loss = train_loss / len(train_loader)
        gap = train_f1 - val_f1
        gap_str = f"│ Gap={gap:+.4f}"
        if gap > 0.05:
            gap_str += " ⚠️ OVERFIT"

        print(f"Epoch {global_epoch}: Train Loss={avg_loss:.4f} "
              f"│ Train F1={train_f1:.4f} │ Val F1={val_f1:.4f} {gap_str} │ Val Acc={val_acc:.2f}%")

        logger.log_metrics("train", {"loss": avg_loss, "F1_Macro": train_f1}, global_epoch)
        val_metrics = {"F1_Macro": val_f1, "Acc": val_acc, "Overfit_Gap": gap}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, global_epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_best(teacher, ema, phase_best, best_f1)
            _save_best(teacher, ema, cfg["teacher_ckpt"], best_f1)

        _save_epoch_ckpt(teacher, ema, ckpt_dir, global_epoch)

    print(f"\nPhase 2 done. Best F1: {best_f1:.4f}")
    logger.close()

    teacher.load_state_dict(torch.load(phase_best, weights_only=False))
    return teacher


# ============================================================
# 全流程 wrapper（向后兼容）
# ============================================================
def train_teacher():
    """训练教师模型全流程（2 阶段）"""
    teacher = train_teacher_phase1()
    teacher = train_teacher_phase2(teacher)
    print(f"\n{'='*60}")
    print(f"✓ 教师模型训练完成（2 阶段）")
    print(f"{'='*60}")
    return teacher


if __name__ == "__main__":
    train_teacher()
