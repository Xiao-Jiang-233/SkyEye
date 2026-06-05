# ============ models/distill_wrapper.py ============
"""
知识蒸馏训练器

支持两种蒸馏模式：
1. 软标签蒸馏：KL(soft_student, soft_teacher) × T²
2. 中间层特征蒸馏：MSE(proj(feat_S), feat_T)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import CONFIG
from utils.logger import TrainLogger


class FeatureProjector(nn.Module):
    """
    1×1 卷积投影层：将 Student 的特征通道数映射到 Teacher 的通道数
    用于中间层特征对齐
    """

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, student_feat, teacher_shape):
        # 空间对齐
        if student_feat.shape[2:] != teacher_shape[2:]:
            student_feat = F.adaptive_avg_pool2d(
                student_feat, output_size=teacher_shape[2:]
            )
        return self.proj(student_feat)


class DistillationTrainer:
    """
    知识蒸馏训练器

    组合损失 = α × KL(soft_S, soft_T) × T² + (1-α) × CE(S, labels) + β × MSE(feat_S, feat_T)

    Args:
        teacher: nn.Module — 教师模型（已训练，将被冻结）
        student: nn.Module — 学生模型（待训练）
        device: torch.device
        cfg: dict — CONFIG 字典
    """

    def __init__(self, teacher, student, device, cfg=None):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.cfg = cfg or CONFIG

        # 冻结教师
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 获取中间层通道信息和 block 索引
        self.teacher_channels = teacher.get_stage_channels()
        self.student_channels = student.get_stage_channels()
        self.kd_indices = self._get_kd_indices()

        # 选择关键 stage 用于特征蒸馏（取最后 N 个 stage）
        n_stages = min(len(self.teacher_channels), len(self.student_channels))
        self.proj_layers = nn.ModuleList([
            FeatureProjector(
                self.student_channels[-(n_stages - i)],
                self.teacher_channels[-(n_stages - i)],
            ).to(device)
            for i in range(n_stages)
        ])

        print(f"Feature KD stages: {n_stages}")
        print(f"KD block indices: {self.kd_indices}")
        print(f"Teacher channels: {self.teacher_channels}")
        print(f"Student channels: {self.student_channels}")

    def _get_kd_indices(self):
        """从 timm feature_info 解析 KD 使用的 block 索引，用于对齐 hook 特征。

        timm feature_info 中 module 字段格式为 'blocks.N'，从中提取 N 作为索引。
        B5 features_only=False 时 entries 无 'index' 字段，需从 module 字符串解析。
        """
        info = getattr(self.teacher.backbone, 'feature_info', None)
        if info is None:
            return list(range(len(self.teacher_channels)))
        if hasattr(info, 'info'):
            info = info.info

        indices = []
        for i, entry in enumerate(info):
            # 优先使用显式 index 字段
            if 'index' in entry:
                indices.append(entry['index'])
            elif 'module' in entry:
                # 从 'blocks.N' 解析 N
                parts = entry['module'].split('.')
                try:
                    indices.append(int(parts[-1]))
                except (ValueError, IndexError):
                    indices.append(i)
            else:
                indices.append(i)

        n = min(len(self.teacher_channels), len(self.student_channels))
        return indices[-n:]

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        计算蒸馏损失

        KL 散度软标签损失 + 交叉熵硬标签损失

        Args:
            student_logits: Tensor (B, C)
            teacher_logits: Tensor (B, C)
            labels: Tensor (B,)

        Returns:
            Tensor: 蒸馏损失标量
        """
        T = self.cfg["kd_temperature"]
        alpha = self.cfg["kd_alpha"]

        # KL 散度软标签损失（T² 缩放保证梯度量级与 CE 一致）
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean',
        ) * (T * T)

        # 交叉熵硬标签损失
        hard_loss = F.cross_entropy(
            student_logits, labels,
            label_smoothing=self.cfg["label_smoothing"],
        )

        return alpha * soft_loss + (1 - alpha) * hard_loss

    def feature_loss(self, student_feats, teacher_feats):
        """
        计算中间层特征对齐损失

        Args:
            student_feats: list[Tensor] — 学生中间层特征
            teacher_feats: list[Tensor] — 教师中间层特征

        Returns:
            Tensor: 特征 MSE 损失（取各 stage 均值）
        """
        if len(self.proj_layers) == 0:
            return 0.0  # 无投影层时跳过特征蒸馏

        feat_loss = 0.0
        for s_feat, t_feat, proj in zip(student_feats, teacher_feats, self.proj_layers):
            projected = proj(s_feat, t_feat.shape)
            feat_loss += F.mse_loss(projected, t_feat.detach())
        return feat_loss / len(self.proj_layers)

    def train_step(self, images, labels, optimizer, scaler):
        """
        单步蒸馏训练

        Returns:
            tuple: (total_loss, kd_loss, feat_loss)
        """
        images, labels = images.to(self.device), labels.to(self.device)

        # 获取 Teacher 输出（无梯度）
        with torch.no_grad():
            teacher_logits, teacher_feats = self.teacher(images, return_features=True)

        # 获取 Student 输出
        with autocast(enabled=self.cfg["fp16"]):
            student_logits, student_feats = self.student(images, return_features=True)

            # 按 feature_info 索引选取特征（跳过通道重复的 block）
            all_teacher = list(teacher_feats.values())
            all_student = list(student_feats.values())
            teacher_feat_list = [all_teacher[i] for i in self.kd_indices]
            student_feat_list = [all_student[i] for i in self.kd_indices]

            # 组合损失
            kd_loss = self.distillation_loss(student_logits, teacher_logits, labels)
            feat_loss = self.feature_loss(student_feat_list, teacher_feat_list)
            total_loss = kd_loss + self.cfg["kd_feature_weight"] * feat_loss

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return total_loss.item(), kd_loss.item(), feat_loss.item()

    def train(self, train_loader, val_loader):
        """
        完整蒸馏训练循环

        Args:
            train_loader: DataLoader
            val_loader: DataLoader

        Returns:
            nn.Module: 蒸馏完成的学生模型（已加载最佳权重）
        """
        # 优化器（student + projection layers）
        student_params = list(self.student.parameters()) + list(self.proj_layers.parameters())
        optimizer = optim.AdamW(student_params, lr=self.cfg["kd_lr"], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg["kd_epochs"])
        scaler = GradScaler(enabled=self.cfg["fp16"])

        # TensorBoard 日志
        logger = TrainLogger(log_dir="results/tb_results/distill", use_tb=self.cfg["use_tb"])

        best_f1 = 0.0
        for epoch in range(self.cfg["kd_epochs"]):
            # --- Train ---
            self.student.train()
            self.proj_layers.train()
            total_loss_avg = 0.0

            pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{self.cfg['kd_epochs']}")
            for images, labels in pbar:
                total, kd, feat = self.train_step(images, labels, optimizer, scaler)
                total_loss_avg += total
                pbar.set_postfix({
                    "total": f"{total:.4f}",
                    "kd": f"{kd:.4f}",
                    "feat": f"{feat:.4f}",
                })

            scheduler.step()

            # --- Validate ---
            val_f1, val_acc, per_class_f1 = self.evaluate(val_loader)
            avg_loss = total_loss_avg / len(train_loader)
            print(f"KD Epoch {epoch+1}: Val F1={val_f1:.4f} | Val Acc={val_acc:.2f}%")

            # TensorBoard 记录（F1 为主监控，含 per-class）
            logger.log_metrics("train", {"loss": avg_loss}, epoch + 1)
            val_metrics = {"F1_Macro": val_f1, "Acc": val_acc}
            for cls_name, cls_f1 in per_class_f1.items():
                val_metrics[f"F1_{cls_name}"] = cls_f1
            logger.log_metrics("val", val_metrics, epoch + 1)
            logger.flush()  # 每轮强制写入磁盘

            # 保存最佳
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.student.state_dict(), self.cfg["distilled_ckpt"])
                print(f"  ✓ Best distilled student saved! F1={best_f1:.4f}")

        logger.close()

        # 加载最佳权重
        self.student.load_state_dict(torch.load(self.cfg["distilled_ckpt"], weights_only=False))
        return self.student

    @torch.no_grad()
    def evaluate(self, loader):
        """验证，返回 (macro_f1, accuracy, per_class_f1_dict)"""
        self.student.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            logits = self.student(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        # 从 val_loader 的底层 ImageFolder 获取类名
        class_names = loader.dataset.dataset.classes
        per_class_f1 = dict(zip(class_names, per_class_f1))
        return f1, acc, per_class_f1
