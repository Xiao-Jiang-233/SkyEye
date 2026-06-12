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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
import os
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

    def __init__(self, teacher, student, device, cfg=None, class_names=None):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.cfg = cfg or CONFIG
        self.class_names = class_names  # 避免 evaluate 中链式耦合 loader.dataset.dataset.classes

        # 自适应 AMP
        self.use_amp = self.cfg["fp16"] and torch.cuda.is_available()
        self.amp_dtype = getattr(torch, self.cfg.get("amp_dtype", "float16")) if self.use_amp else None
        self.use_grad_scaler = self.cfg.get("use_grad_scaler", False) and self.use_amp
        self.scaler = torch.amp.GradScaler('cuda') if self.use_grad_scaler else None

        # 冻结教师
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 获取中间层通道信息（兼容 DataParallel 包装）
        teacher_model = teacher.module if isinstance(teacher, nn.DataParallel) else teacher
        student_model = student.module if isinstance(student, nn.DataParallel) else student
        self.teacher_channels = teacher_model.get_stage_channels()
        self.student_channels = student_model.get_stage_channels()
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
        B4 features_only=False 时 entries 无 'index' 字段，需从 module 字符串解析。
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
            return torch.tensor(0.0, device=self.device)  # 无投影层时跳过特征蒸馏

        feat_loss = 0.0
        for s_feat, t_feat, proj in zip(student_feats, teacher_feats, self.proj_layers):
            projected = proj(s_feat, t_feat.shape)
            feat_loss += F.mse_loss(projected, t_feat.detach())
        return feat_loss / len(self.proj_layers)

    def train_step(self, images, labels, optimizer):
        """
        单步蒸馏训练（自适应 AMP：bfloat16 或 float16 + GradScaler）

        Returns:
            tuple: (total_loss, kd_loss, feat_loss, preds)
        """
        images, labels = images.to(self.device), labels.to(self.device)

        # 获取 Teacher 输出（无梯度）
        with torch.no_grad():
            teacher_logits, teacher_feats = self.teacher(images, return_features=True)

        # 获取 Student 输出
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
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
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        preds = student_logits.detach().argmax(dim=1)
        return total_loss.item(), kd_loss.item(), feat_loss.item(), preds

    def train(self, train_loader, val_loader):
        """
        完整蒸馏训练循环

        每个 epoch 保存 per-epoch checkpoint（保留最近 20 个），每轮刷新 TensorBoard。
        监控 overfit gap（train_f1 - val_f1），超过 0.05 会警告。

        Args:
            train_loader: DataLoader
            val_loader: DataLoader

        Returns:
            nn.Module: 蒸馏完成的学生模型（已加载最佳权重）
        """
        ckpt_dir = self.cfg.get("distill_ckpt_dir", "results/checkpoints/distill")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 优化器（student + projection layers）
        student_params = list(self.student.parameters()) + list(self.proj_layers.parameters())
        optimizer = optim.AdamW(student_params, lr=self.cfg["kd_lr"], weight_decay=1e-4)

        # 调度器：Linear warmup → CosineAnnealing
        warmup_epochs = self.cfg.get("warmup_epochs", 2)
        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=self.cfg["kd_epochs"] - warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg["kd_epochs"])

        # TensorBoard 日志
        logger = TrainLogger(log_dir="results/tb_results/distill", use_tb=self.cfg["use_tb"])

        best_f1 = 0.0
        for epoch in range(self.cfg["kd_epochs"]):
            global_epoch = epoch + 1

            # --- Train ---
            self.student.train()
            self.proj_layers.train()
            total_loss_avg = 0.0
            train_preds, train_labels_list = [], []

            pbar = tqdm(train_loader, desc=f"KD Epoch {global_epoch}/{self.cfg['kd_epochs']}")
            for images, labels in pbar:
                total, kd, feat, preds = self.train_step(images, labels, optimizer)
                total_loss_avg += total

                # 收集训练预测（train_step 中已完成前向，直接复用其 logits）
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    "total": f"{total:.4f}",
                    "kd": f"{kd:.4f}",
                    "feat": f"{feat:.4f}",
                })

            scheduler.step()

            # Train F1
            train_f1 = f1_score(train_labels_list, train_preds, average='macro')

            # --- Validate ---
            val_f1, val_acc, per_class_f1 = self.evaluate(val_loader)
            avg_loss = total_loss_avg / len(train_loader)

            # Overfit gap
            gap = train_f1 - val_f1
            gap_str = f"| Gap={gap:+.4f}"
            if gap > 0.05:
                gap_str += " ⚠️ OVERFIT"

            print(f"KD Epoch {global_epoch}: Train Loss={avg_loss:.4f} "
                  f"| Train F1={train_f1:.4f} | Val F1={val_f1:.4f} {gap_str} | Val Acc={val_acc:.2f}%")

            # TensorBoard 记录（F1 为主监控，含 per-class）
            logger.log_metrics("train", {"loss": avg_loss, "F1_Macro": train_f1}, global_epoch)
            val_metrics = {"F1_Macro": val_f1, "Acc": val_acc, "Overfit_Gap": gap}
            for cls_name, cls_f1 in per_class_f1.items():
                val_metrics[f"F1_{cls_name}"] = cls_f1
            logger.log_metrics("val", val_metrics, global_epoch)
            logger.flush()  # 每轮强制写入磁盘

            # 保存最佳
            if val_f1 > best_f1:
                best_f1 = val_f1
                save_model = self.student.module if isinstance(self.student, nn.DataParallel) else self.student
                torch.save(save_model.state_dict(), self.cfg["distilled_ckpt"])
                print(f"  ✓ Best distilled student saved! F1={best_f1:.4f}")

            # 每 epoch 周期备份（保留最近 20 个）
            ckpt_path = os.path.join(ckpt_dir, f"distill_epoch_{global_epoch:02d}.pth")
            save_model = self.student.module if isinstance(self.student, nn.DataParallel) else self.student
            torch.save(save_model.state_dict(), ckpt_path)
            old = os.path.join(ckpt_dir, f"distill_epoch_{global_epoch-20:02d}.pth")
            if os.path.exists(old):
                os.remove(old)

        logger.close()

        # 加载最佳权重
        s = self.student.module if isinstance(self.student, nn.DataParallel) else self.student
        s.load_state_dict(torch.load(self.cfg["distilled_ckpt"], weights_only=False))
        return self.student

    def _build_logit_bias(self):
        """从 config 构建 per-class logit bias 张量（方案 A）。
        实现为 logits - bias，因此 bias > 0 → 更难被预测，bias < 0 → 更容易被预测。"""
        bias = torch.zeros(len(self.class_names), device=self.device)
        bias_cfg = self.cfg.get("logit_bias", {})
        if bias_cfg:
            for cls_name, val in bias_cfg.items():
                if cls_name in self.class_names:
                    bias[self.class_names.index(cls_name)] = val
        return bias

    @torch.no_grad()
    def evaluate(self, loader):
        """验证，返回 (macro_f1, accuracy, per_class_f1_dict)"""
        self.student.eval()
        logit_bias = self._build_logit_bias()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            logits = self.student(images)
            logits = logits - logit_bias.unsqueeze(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        per_class_f1 = dict(zip(self.class_names, per_class_f1))
        return f1, acc, per_class_f1
