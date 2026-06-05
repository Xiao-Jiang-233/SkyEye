# ============ training/prune_finetune.py ============
"""
结构化剪枝 + 渐进微调

策略：
1. 对 MBConv 中 1×1 点卷积进行 L2-norm 通道剪枝
2. 渐进式 3 轮剪枝（15% → 27.5% → 40%），每轮后微调恢复精度
3. 最终固化 mask 生成密集小模型
输出：results/student_pruned_final.pth
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
import numpy as np
from tqdm import tqdm
from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.dataset import create_dataloaders, compute_class_weights
from training.train_teacher import FocalLoss, evaluate
from utils.logger import TrainLogger


class StructuredPruner:
    """
    EfficientNet 结构化通道剪枝器

    只剪枝 MBConv 中的 1×1 点卷积层（通道操作核心），
    跳过输入层（≤32 channels）和深度可分离卷积。

    Args:
        model: WeatherEfficientNet
        prune_ratio: float — 剪枝比例
        method: str — 'l2' / 'l1'
    """

    def __init__(self, model, prune_ratio=0.4, method='l2'):
        self.model = model
        self.prune_ratio = prune_ratio
        self.method = method
        self.pruned_params = []

    def apply_pruning(self):
        """对所有 1×1 点卷积应用结构化剪枝"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                # 跳过输入层和输出层附近的小通道卷积
                if module.in_channels <= 32 or module.out_channels <= 32:
                    continue

                norm_type = 2 if self.method == 'l2' else 1
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.prune_ratio,
                    n=norm_type,
                    dim=0,  # 剪输出通道
                )
                self.pruned_params.append((name, 'output_channels'))

        self._print_stats()

    def _print_stats(self):
        """打印剪枝统计"""
        total_weights = 0
        zero_weights = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_mask'):
                total_weights += module.weight_mask.numel()
                zero_weights += (module.weight_mask == 0).sum().item()

        sparsity = zero_weights / total_weights * 100 if total_weights > 0 else 0
        print(f"\n{'='*50}")
        print(f"Pruning Statistics:")
        print(f"  Total weighted params: {total_weights:,}")
        print(f"  Zero (pruned) params:  {zero_weights:,}")
        print(f"  Structured Sparsity:   {sparsity:.2f}%")
        print(f"  Layers pruned:         {len(self.pruned_params)}")
        print(f"{'='*50}\n")

    @staticmethod
    def make_permanent(model):
        """将 mask 固化到权重中（静态方法，无需实例化）"""
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')


def finetune_after_prune(model, train_loader, val_loader, class_counts, class_names, device, cfg, epochs, lr, tag=""):
    """
    剪枝后微调：仅更新未剪枝的权重

    Args:
        model: 已剪枝的模型
        train_loader: DataLoader
        val_loader: DataLoader
        class_counts: np.ndarray — 各类别样本数（由 create_dataloaders 返回，避免重复遍历）
        class_names: list[str] — 类别名称列表
        device: torch.device
        cfg: dict
        epochs: int
        lr: float
        tag: str — 日志标签

    Returns:
        nn.Module: 微调后的模型
    """
    alpha = compute_class_weights(class_counts)
    criterion = FocalLoss(
        alpha=alpha, gamma=cfg["focal_gamma"],
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 调度器：Linear warmup → CosineAnnealing
    warmup_epochs = cfg.get("warmup_epochs", 2)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard 日志（每轮微调独立目录）
    log_dir = f"results/tb_results/prune_{tag}" if tag else "results/tb_results/prune"
    logger = TrainLogger(log_dir=log_dir, use_tb=cfg["use_tb"])

    ckpt_path = f"results/student_pruned_{tag}.pth" if tag else "results/student_pruned_temp.pth"
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"{tag} FT Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16, enabled=cfg["fp16"]):
                loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validate（复用 train_teacher 的 evaluate 函数）
        f1, acc, per_class_f1 = evaluate(model, val_loader, device, class_names)
        avg_loss = train_loss / len(train_loader)
        print(f"  [{tag}] Epoch {epoch+1}: F1={f1:.4f}, Acc={acc:.2f}%")

        # TensorBoard 记录（F1 为主监控，含 per-class）
        logger.log_metrics("train", {"loss": avg_loss}, epoch + 1)
        val_metrics = {"F1_Macro": f1, "Acc": acc}
        for cls_name, cls_f1 in per_class_f1.items():
            val_metrics[f"F1_{cls_name}"] = cls_f1
        logger.log_metrics("val", val_metrics, epoch + 1)
        logger.flush()  # 每轮强制写入磁盘

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)

    logger.close()

    model.load_state_dict(torch.load(ckpt_path, weights_only=False))
    return model


def prune_and_finetune():
    """
    完整剪枝 + 渐进微调流水线

    流程：加载蒸馏模型 → 3 轮渐进剪枝 → 最终微调 → 固化
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 1) 创建学生模型并加载蒸馏权重
    student = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    student.load_state_dict(torch.load(cfg["distilled_ckpt"], weights_only=False))
    print(f"Distilled student loaded from {cfg['distilled_ckpt']}")

    # 2) 数据加载
    train_loader, val_loader, class_counts, class_names = create_dataloaders()

    # 3) 渐进式剪枝
    ratios = np.linspace(0.15, cfg["prune_ratio"], cfg["prune_iterations"])

    for i, ratio in enumerate(ratios):
        print(f"\n{'#'*50}")
        print(f"# Pruning Iteration {i+1}/{cfg['prune_iterations']} (ratio={ratio:.2%})")
        print(f"{'#'*50}")

        pruner = StructuredPruner(student, prune_ratio=ratio, method='l2')
        pruner.apply_pruning()

        student = finetune_after_prune(
            student, train_loader, val_loader, class_counts, class_names, device, cfg,
            epochs=cfg["prune_finetune_epochs"],
            lr=cfg["prune_finetune_lr"],
            tag=f"iter{i+1}",
        )

    # 4) 固化剪枝
    StructuredPruner.make_permanent(student)
    print("✓ Pruning masks merged into weights (permanent)")

    # 5) 最终微调
    print("\nFinal fine-tuning...")
    student = finetune_after_prune(
        student, train_loader, val_loader, device, cfg,
        epochs=cfg["prune_finetune_epochs"],
        lr=cfg["prune_finetune_lr"] * 0.5,
        tag="final",
    )

    # 6) 保存最终模型
    torch.save(student.state_dict(), cfg["pruned_ckpt"])
    print(f"✓ Final pruned model saved to {cfg['pruned_ckpt']}")

    # 7) 压缩率对比
    import timm
    original = timm.create_model(cfg["student_model"], pretrained=False, num_classes=cfg["num_classes"])
    orig_params = sum(p.numel() for p in original.parameters())
    pruned_params = sum(p.numel() for p in student.parameters())

    print(f"\n{'='*50}")
    print(f"Compression Summary:")
    print(f"  Original params:  {orig_params:>12,}")
    print(f"  Pruned params:    {pruned_params:>12,}")
    print(f"  Compression:      {pruned_params/orig_params*100:.1f}%")
    print(f"  Reduction:        {(1-pruned_params/orig_params)*100:.1f}%")
    print(f"{'='*50}")

    return student


if __name__ == "__main__":
    prune_and_finetune()
