# EfficientNet + 知识蒸馏 + 结构化剪枝 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零构建完整的天气分类管线：EfficientNet-B5 教师训练 → B0 知识蒸馏 → 结构化剪枝 → ONNX 导出 → INT8 量化 → CPU 推理

**Architecture:** 纯模块化 .py 项目结构。config.py 统一管理超参数，data/ 处理数据加载增强，models/ 封装网络和蒸馏逻辑，training/ 实现三阶段训练流水线，inference/ 负责 ONNX 导出和推理。coding_here.ipynb 作为 Mo 平台入口分阶段调用各模块。

**Tech Stack:** PyTorch, timm, ONNX Runtime, scikit-learn, tqdm, Mo 平台 (JupyterLab + GPU)

**Design Doc:** `docs/superpowers/specs/2026-06-03-efficientnet-kd-pruning-design.md`

---

## 文件结构

```
SkyEye/
├── config.py                      # 创建: 超参数配置
├── utils/
│   ├── __init__.py               # 创建: 空文件
│   ├── metrics.py                 # 创建: F1 / 混淆矩阵 / 分类报告
│   └── logger.py                  # 创建: 训练日志 + TensorBoard
├── data/
│   ├── __init__.py               # 创建: 空文件
│   ├── augmentations.py           # 创建: train/val 增强策略
│   └── dataset.py                 # 创建: ImageFolder 加载 + 类别权重计算
├── models/
│   ├── __init__.py               # 创建: 空文件
│   ├── weather_efficientnet.py    # 创建: EfficientNet 封装 + 中间层 hook
│   └── distill_wrapper.py         # 创建: 蒸馏损失 + 特征投影 + 训练循环
├── training/
│   ├── __init__.py               # 创建: 空文件
│   ├── train_teacher.py           # 创建: 教师模型训练
│   ├── distill_student.py         # 创建: 知识蒸馏入口
│   └── prune_finetune.py          # 创建: 结构化剪枝 + 渐进微调
├── inference/
│   ├── __init__.py               # 创建: 空文件
│   ├── export_onnx.py             # 创建: ONNX 导出 + 简化 + 推理测速
│   └── infer.py                   # 创建: 单张/批量推理
├── results/                       # 已存在: 模型权重存放
└── coding_here.ipynb              # 修改: Notebook 入口（分阶段调用）
```

---

### Task 1: 超参数配置文件

**Files:**
- Create: `config.py`

- [ ] **Step 1: 创建 config.py**

```python
# ============ config.py ============
"""
SkyEye 天气分类项目 — 超参数配置中心
所有模块通过 `from config import CONFIG` 统一获取参数
"""
import torch

CONFIG = {
    # ---- 数据 ----
    "data_root": "datasets/69f46e75dbb43ba9e05483c1-69e0f1d5638ba61f00d54c83/weather_classification",
    "writable_root": ".data/weather",  # 将只读数据集复制到此可写目录
    "num_classes": 6,
    "class_names": ["cloudy", "haze", "rainy", "snow", "sunny", "thunder"],
    "img_size": 224,               # EfficientNet 标准输入
    "batch_size": 32,
    "val_split": 0.15,             # 验证集比例

    # ---- 教师模型 ----
    "teacher_model": "efficientnet-b5",  # timm 模型名
    "teacher_pretrained": True,
    "teacher_epochs": 30,
    "teacher_lr": 1e-3,
    "teacher_weight_decay": 1e-4,

    # ---- 知识蒸馏 ----
    "student_model": "efficientnet-b0",  # timm 模型名
    "kd_temperature": 4.0,               # 蒸馏温度 T
    "kd_alpha": 0.7,                     # 软标签损失权重
    "kd_feature_weight": 0.1,            # 中间层特征损失权重
    "kd_epochs": 40,
    "kd_lr": 1e-3,

    # ---- 结构化剪枝 ----
    "prune_ratio": 0.4,           # 最终剪枝比例
    "prune_iterations": 3,        # 渐进剪枝轮数
    "prune_finetune_epochs": 15,  # 剪枝后微调轮数
    "prune_finetune_lr": 1e-4,    # 微调学习率（比训练时更低）

    # ---- 通用 ----
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "fp16": True,                 # 混合精度训练
    "num_workers": 4,
    "scheduler": "cosine",        # cosine / plateau
    "label_smoothing": 0.1,
    "use_focal_loss": True,       # 处理类别不平衡
    "focal_gamma": 2.0,

    # ---- 路径 ----
    "teacher_ckpt": "results/teacher_best.pth",
    "distilled_ckpt": "results/student_distilled_best.pth",
    "pruned_ckpt": "results/student_pruned_final.pth",
    "onnx_path": "results/weather_model.onnx",
}
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from config import CONFIG; print(f'Device: {CONFIG[\"device\"]}'); print(f'Classes: {CONFIG[\"class_names\"]}')"
```

- [ ] **Step 3: 提交**

```bash
git add config.py
git commit -m "feat: add config.py with all hyperparameters"
```

---

### Task 2: 工具模块 — 指标和日志

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/metrics.py`
- Create: `utils/logger.py`

- [ ] **Step 1: 创建 __init__.py**

```bash
mkdir -p utils && touch utils/__init__.py
```

- [ ] **Step 2: 创建 utils/metrics.py**

```python
# ============ utils/metrics.py ============
"""模型评估指标：Macro F1、混淆矩阵、分类报告"""
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def compute_metrics(all_labels, all_preds, class_names):
    """
    计算多分类评估指标

    Args:
        all_labels: list[int] — 真实标签列表
        all_preds: list[int] — 预测标签列表
        class_names: list[str] — 类别名称列表

    Returns:
        dict: {f1, accuracy, confusion_matrix, report}
    """
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    f1 = f1_score(labels_arr, preds_arr, average='macro')
    accuracy = (preds_arr == labels_arr).mean() * 100
    cm = confusion_matrix(labels_arr, preds_arr)
    report = classification_report(
        labels_arr, preds_arr,
        target_names=class_names,
        digits=4,
    )

    return {
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
    }


def print_metrics(metrics):
    """格式化打印评估指标"""
    print(f"\n{'='*50}")
    print(f"Macro F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"{'='*50}")
    print("Classification Report:")
    print(metrics["report"])
    print(f"{'='*50}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"{'='*50}\n")
```

- [ ] **Step 3: 创建 utils/logger.py**

```python
# ============ utils/logger.py ============
"""训练日志工具：控制台输出 + TensorBoard（可选）"""
import os
import time
from datetime import datetime


class TrainLogger:
    """简单的训练日志记录器，支持 TensorBoard SummaryWriter"""

    def __init__(self, log_dir=None, use_tb=False):
        self.use_tb = use_tb
        self.writer = None

        if use_tb:
            from torch.utils.tensorboard import SummaryWriter
            if log_dir is None:
                log_dir = f"results/tb_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        self.start_time = time.time()
        self.metrics_history = {}

    def log_scalar(self, tag, value, step):
        """记录标量值（TensorBoard）"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, phase, metrics, epoch):
        """
        记录一个 epoch 的指标集合

        Args:
            phase: str — 'train' / 'val' / 'test'
            metrics: dict — 指标字典
            epoch: int — 当前 epoch
        """
        key = f"{phase}_{epoch}"
        self.metrics_history[key] = metrics

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{phase}/{name}", value, epoch)

    def elapsed(self):
        """返回已用时间（秒）"""
        return time.time() - self.start_time

    def close(self):
        if self.writer:
            self.writer.close()
```

- [ ] **Step 4: 验证导入**

```bash
python -c "from utils.metrics import compute_metrics, print_metrics; from utils.logger import TrainLogger; print('utils OK')"
```

- [ ] **Step 5: 提交**

```bash
git add utils/
git commit -m "feat: add metrics and logger utility modules"
```

---

### Task 3: 数据模块 — 增强策略

**Files:**
- Create: `data/__init__.py`
- Create: `data/augmentations.py`

- [ ] **Step 1: 创建 data/__init__.py**

```bash
mkdir -p data && touch data/__init__.py
```

- [ ] **Step 2: 创建 data/augmentations.py**

```python
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
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from data.augmentations import get_train_transforms, get_val_transforms; t = get_train_transforms(); print(f'Train transforms: {len(t.transforms)} ops'); v = get_val_transforms(); print(f'Val transforms: {len(v.transforms)} ops')"
```

- [ ] **Step 4: 提交**

```bash
git add data/
git commit -m "feat: add data augmentation transforms"
```

---

### Task 4: 数据模块 — 数据集加载

**Files:**
- Modify: `data/dataset.py`

- [ ] **Step 1: 创建 data/dataset.py**

```python
# ============ data/dataset.py ============
"""数据集加载：ImageFolder → DataLoader，含类别权重计算"""
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CONFIG
from data.augmentations import get_train_transforms, get_val_transforms


def prepare_data():
    """
    将 datasets/ 只读数据复制到 .data/weather/ 可写目录（仅首次运行）

    Returns:
        str: 可写数据目录路径
    """
    src = CONFIG["data_root"]
    dst = CONFIG["writable_root"]

    if not os.path.exists(dst):
        print(f"Copying dataset from {src} to {dst} ...")
        shutil.copytree(src, dst)
        print("Dataset copy complete.")
    else:
        print(f"Dataset already exists at {dst}")

    return dst


def create_dataloaders(data_root=None, img_size=None, batch_size=None, num_workers=None):
    """
    创建训练和验证 DataLoader

    Args:
        data_root: str — 数据根目录（默认从 CONFIG 读取）
        img_size: int — 图片尺寸
        batch_size: int — 批次大小
        num_workers: int — 数据加载线程数

    Returns:
        tuple: (train_loader, val_loader, class_counts)
    """
    cfg = CONFIG
    root = data_root or prepare_data()
    size = img_size or cfg["img_size"]
    bs = batch_size or cfg["batch_size"]
    nw = num_workers or cfg["num_workers"]

    # 全量加载以获取类别分布
    full_dataset = ImageFolder(root)
    num_classes = len(full_dataset.classes)
    class_counts = np.bincount(full_dataset.targets, minlength=num_classes)

    # 分层划分训练集/验证集
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg["val_split"],
        stratify=full_dataset.targets,
        random_state=cfg["seed"],
    )

    # 分别创建两个 ImageFolder 实例（不同 transform）+ Subset
    train_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_train_transforms(size)),
        train_idx,
    )
    val_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_val_transforms(size)),
        val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    print(f"Classes: {full_dataset.classes}")
    print(f"Class distribution: {dict(zip(full_dataset.classes, class_counts.astype(int)))}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_loader, val_loader, class_counts


def compute_class_weights(class_counts):
    """
    根据类别样本数计算 FocalLoss 的 alpha 参数

    Args:
        class_counts: np.ndarray — 各类别样本数

    Returns:
        torch.Tensor: 归一化的 alpha 权重
    """
    alpha = 1.0 / (class_counts + 1e-8)
    alpha = alpha / alpha.sum() * len(class_counts)
    return torch.tensor(alpha, dtype=torch.float32)
```

- [ ] **Step 2: 验证导入（无需真实数据，检查语法）**

```bash
python -c "from data.dataset import create_dataloaders, compute_class_weights; print('dataset module OK')"
```

- [ ] **Step 3: 提交**

```bash
git add data/dataset.py
git commit -m "feat: add dataset loader with stratified split and class weights"
```

---

### Task 5: 模型模块 — EfficientNet 封装

**Files:**
- Create: `models/__init__.py`
- Create: `models/weather_efficientnet.py`

- [ ] **Step 1: 创建 models/__init__.py**

```bash
mkdir -p models && touch models/__init__.py
```

- [ ] **Step 2: 创建 models/weather_efficientnet.py**

```python
# ============ models/weather_efficientnet.py ============
"""EfficientNet 天气分类模型，支持中间层特征提取用于知识蒸馏"""
import timm
import torch.nn as nn
import torch.nn.functional as F


class WeatherEfficientNet(nn.Module):
    """
    EfficientNet 天气分类器

    使用 timm 库加载预训练 EfficientNet，去除原始分类头，
    替换为自定义分类器。通过 forward hook 捕获中间层特征，
    用于知识蒸馏中的特征对齐。

    Args:
        model_name: str — timm 模型名 (e.g. "efficientnet-b0", "efficientnet-b5")
        num_classes: int — 分类类别数
        pretrained: bool — 是否加载 ImageNet 预训练权重
    """

    def __init__(self, model_name="efficientnet-b0", num_classes=6, pretrained=True):
        super().__init__()
        # 加载 timm EfficientNet，去掉分类头，保留空间特征
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # 去掉分类头
            global_pool='',      # 保留空间特征图，不做全局池化
            features_only=False,
        )

        # 获取 backbone 输出通道数 (B0=1280, B5=2048)
        self.num_features = self.backbone.num_features

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # 中间层特征存储
        self.intermediate_features = {}

        # 注册 hook 捕获各 stage 输出
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hook 收集中间层特征"""

        def hook_fn(name):
            def fn(_, __, output):
                self.intermediate_features[name] = output
            return fn

        for i, block in enumerate(self.backbone.blocks):
            block.register_forward_hook(hook_fn(f"stage_{i}"))

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: Tensor (B,3,H,W)
            return_features: bool — 是否返回中间层特征（KD 时需要）

        Returns:
            if return_features:
                (logits, intermediate_features_dict)
            else:
                logits
        """
        self.intermediate_features = {}

        # Backbone 前向
        x = self.backbone.forward_features(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # 分类
        logits = self.classifier(x)

        if return_features:
            return logits, self.intermediate_features
        return logits

    def get_stage_channels(self):
        """
        返回各 stage 的输出通道数，用于特征蒸馏时的投影层配置

        Returns:
            list[int]: 各 stage 的输出通道数（去重后）
        """
        channels = []
        prev_channels = None

        for block in self.backbone.blocks:
            # MBConv 模块的点卷积输出通道
            if hasattr(block, 'conv_pwl'):
                ch = block.conv_pwl.out_channels
            elif hasattr(block, 'conv_pw'):
                ch = block.conv_pw.out_channels
            else:
                continue

            if ch != prev_channels:
                channels.append(ch)
                prev_channels = ch

        return channels
```

- [ ] **Step 3: 验证导入和模型构建**

```bash
python -c "from models.weather_efficientnet import WeatherEfficientNet; m = WeatherEfficientNet('efficientnet-b0', num_classes=6); print(f'Features: {m.num_features}'); print(f'Stages: {len(m.get_stage_channels())}')"
```

- [ ] **Step 4: 提交**

```bash
git add models/
git commit -m "feat: add EfficientNet model wrapper with intermediate hooks"
```

---

### Task 6: 模型模块 — 知识蒸馏包装器

**Files:**
- Create: `models/distill_wrapper.py`

- [ ] **Step 1: 创建 models/distill_wrapper.py**

```python
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

        # 获取中间层通道信息
        self.teacher_channels = teacher.get_stage_channels()
        self.student_channels = student.get_stage_channels()

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
        print(f"Teacher channels: {self.teacher_channels}")
        print(f"Student channels: {self.student_channels}")

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

            # 取最后 N 个 stage 的特征
            teacher_feat_list = list(teacher_feats.values())[-len(self.proj_layers):]
            student_feat_list = list(student_feats.values())[-len(self.proj_layers):]

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
            val_f1, val_acc = self.evaluate(val_loader)
            print(f"KD Epoch {epoch+1}: Val F1={val_f1:.4f} | Val Acc={val_acc:.2f}%")

            # 保存最佳
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.student.state_dict(), self.cfg["distilled_ckpt"])
                print(f"  ✓ Best distilled student saved! F1={best_f1:.4f}")

        # 加载最佳权重
        self.student.load_state_dict(torch.load(self.cfg["distilled_ckpt"]))
        return self.student

    @torch.no_grad()
    def evaluate(self, loader):
        """验证"""
        self.student.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device)
            logits = self.student(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        return f1, acc
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from models.distill_wrapper import DistillationTrainer, FeatureProjector; print('distill_wrapper OK')"
```

- [ ] **Step 3: 提交**

```bash
git add models/distill_wrapper.py
git commit -m "feat: add distillation trainer with soft-label and feature alignment"
```

---

### Task 7: 训练模块 — 教师模型训练

**Files:**
- Create: `training/__init__.py`
- Create: `training/train_teacher.py`

- [ ] **Step 1: 创建 training/__init__.py**

```bash
mkdir -p training && touch training/__init__.py
```

- [ ] **Step 2: 创建 training/train_teacher.py**

```python
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
def evaluate(model, loader, device):
    """
    在验证集上评估模型

    Returns:
        tuple: (macro_f1, accuracy)
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
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return f1, acc


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
        val_f1, val_acc = evaluate(teacher, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f} | Val F1={val_f1:.4f} | Val Acc={val_acc:.2f}%")

        # 保存最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(teacher.state_dict(), cfg["teacher_ckpt"])
            print(f"  ✓ Best teacher saved! F1={best_f1:.4f}")

    print(f"\nTeacher training done. Best F1: {best_f1:.4f}")

    # 加载最佳权重
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"]))
    return teacher


if __name__ == "__main__":
    train_teacher()
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from training.train_teacher import train_teacher, FocalLoss; print('train_teacher OK')"
```

- [ ] **Step 4: 提交**

```bash
git add training/
git commit -m "feat: add teacher training script with FocalLoss and mixed precision"
```

---

### Task 8: 训练模块 — 知识蒸馏入口

**Files:**
- Create: `training/distill_student.py`

- [ ] **Step 1: 创建 training/distill_student.py**

```python
# ============ training/distill_student.py ============
"""
知识蒸馏入口脚本

流程：加载 Teacher → 创建 Student → 初始化 DistillationTrainer → 训练
输出：results/student_distilled_best.pth
"""
import torch

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from models.distill_wrapper import DistillationTrainer
from data.dataset import create_dataloaders


def run_distillation():
    """执行完整知识蒸馏流程"""
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 1) 加载训练好的教师模型
    print(f"Loading teacher: {cfg['teacher_model']} ...")
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,  # 使用自己训练的权重
    ).to(device)
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"]))
    teacher.eval()
    print("Teacher model loaded and frozen.")

    # 2) 创建学生模型
    print(f"Creating student: {cfg['student_model']} ...")
    student = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=True,  # ImageNet 预训练
    ).to(device)
    print(f"Student model created.")

    # 3) 数据加载
    train_loader, val_loader, _ = create_dataloaders()

    # 4) 蒸馏训练
    trainer = DistillationTrainer(teacher, student, device, cfg)
    distilled_student = trainer.train(train_loader, val_loader)

    print(f"Distillation complete. Model saved to {cfg['distilled_ckpt']}")
    return distilled_student


if __name__ == "__main__":
    run_distillation()
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from training.distill_student import run_distillation; print('distill_student OK')"
```

- [ ] **Step 3: 提交**

```bash
git add training/distill_student.py
git commit -m "feat: add distillation entry script"
```

---

### Task 9: 训练模块 — 结构化剪枝与微调

**Files:**
- Create: `training/prune_finetune.py`

- [ ] **Step 1: 创建 training/prune_finetune.py**

```python
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
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.dataset import create_dataloaders, compute_class_weights
from training.train_teacher import FocalLoss


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

    def make_permanent(self):
        """将 mask 固化到权重中"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')


def finetune_after_prune(model, train_loader, val_loader, device, cfg, epochs, lr, tag=""):
    """
    剪枝后微调：仅更新未剪枝的权重

    Args:
        model: 已剪枝的模型
        train_loader: DataLoader
        val_loader: DataLoader
        device: torch.device
        cfg: dict
        epochs: int
        lr: float
        tag: str — 日志标签

    Returns:
        nn.Module: 微调后的模型
    """
    # 计算类别权重
    class_counts = np.zeros(cfg["num_classes"])
    for _, labels in train_loader:
        for lbl in labels.numpy():
            class_counts[lbl] += 1

    alpha = compute_class_weights(class_counts)
    criterion = FocalLoss(alpha=alpha, gamma=cfg["focal_gamma"])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=cfg["fp16"])

    ckpt_path = f"results/student_pruned_{tag}.pth" if tag else "results/student_pruned_temp.pth"
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"{tag} FT Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg["fp16"]):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        print(f"  [{tag}] Epoch {epoch+1}: F1={f1:.4f}, Acc={acc:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
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
    student.load_state_dict(torch.load(cfg["distilled_ckpt"]))
    print(f"Distilled student loaded from {cfg['distilled_ckpt']}")

    # 2) 数据加载
    train_loader, val_loader, _ = create_dataloaders()

    # 3) 渐进式剪枝
    ratios = np.linspace(0.15, cfg["prune_ratio"], cfg["prune_iterations"])

    for i, ratio in enumerate(ratios):
        print(f"\n{'#'*50}")
        print(f"# Pruning Iteration {i+1}/{cfg['prune_iterations']} (ratio={ratio:.2%})")
        print(f"{'#'*50}")

        pruner = StructuredPruner(student, prune_ratio=ratio, method='l2')
        pruner.apply_pruning()

        student = finetune_after_prune(
            student, train_loader, val_loader, device, cfg,
            epochs=cfg["prune_finetune_epochs"],
            lr=cfg["prune_finetune_lr"],
            tag=f"iter{i+1}",
        )

    # 4) 固化剪枝
    final_pruner = StructuredPruner(student, prune_ratio=0, method='l2')
    final_pruner.make_permanent()
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
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from training.prune_finetune import prune_and_finetune, StructuredPruner; print('prune_finetune OK')"
```

- [ ] **Step 3: 提交**

```bash
git add training/prune_finetune.py
git commit -m "feat: add structured pruning with progressive fine-tuning"
```

---

### Task 10: 推理模块 — ONNX 导出与推理

**Files:**
- Create: `inference/__init__.py`
- Create: `inference/export_onnx.py`
- Create: `inference/infer.py`

- [ ] **Step 1: 创建 inference/__init__.py**

```bash
mkdir -p inference && touch inference/__init__.py
```

- [ ] **Step 2: 创建 inference/export_onnx.py**

```python
# ============ inference/export_onnx.py ============
"""
ONNX 模型导出 + 图简化 + PyTorch vs ONNX Runtime 推理测速

输出：results/weather_model.onnx
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
import time

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet


def export_to_onnx(model_path, onnx_path=None):
    """
    将 PyTorch 模型导出为 ONNX 格式

    Args:
        model_path: str — .pth 权重文件路径
        onnx_path: str — 输出 .onnx 路径

    Returns:
        str: ONNX 文件路径
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])
    onnx_path = onnx_path or cfg["onnx_path"]

    # 加载模型
    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建 dummy input
    dummy_input = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model exported and validated: {onnx_path}")

    # 尝试简化图结构
    try:
        import onnxsim
        model_simp, check = onnxsim.simplify(onnx_path)
        if check:
            onnx.save(model_simp, onnx_path)
            print("✓ ONNX model simplified")
        else:
            print("! onnxsim simplification check failed, keeping original")
    except ImportError:
        print("! onnxsim not installed, skipping simplification (pip install onnxsim)")

    return onnx_path


def benchmark_inference(model_path, onnx_path=None):
    """
    对比 PyTorch 原生 vs ONNX Runtime 推理速度

    Args:
        model_path: str — .pth 权重文件
        onnx_path: str — .onnx 文件（如已有则跳过导出）
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])

    # --- PyTorch Inference ---
    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)

    # Warmup
    for _ in range(20):
        _ = model(dummy)

    # Benchmark
    torch_times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_times.append(time.perf_counter() - start)

    torch_avg = np.mean(torch_times) * 1000  # ms
    torch_std = np.std(torch_times) * 1000
    print(f"PyTorch inference: {torch_avg:.3f} ± {torch_std:.3f} ms")

    # --- ONNX Runtime Inference ---
    if onnx_path is None:
        onnx_path = cfg["onnx_path"]

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    dummy_np = dummy.cpu().numpy()

    # Warmup
    for _ in range(20):
        _ = session.run(None, {'input': dummy_np})

    # Benchmark
    onnx_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = session.run(None, {'input': dummy_np})
        onnx_times.append(time.perf_counter() - start)

    onnx_avg = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000
    speedup = torch_avg / onnx_avg
    print(f"ONNX Runtime inference: {onnx_avg:.3f} ± {onnx_std:.3f} ms")
    print(f"Speedup: {speedup:.2f}×")

    return {"torch_ms": torch_avg, "onnx_ms": onnx_avg, "speedup": speedup}


if __name__ == "__main__":
    onnx_path = export_to_onnx(CONFIG["pruned_ckpt"])
    benchmark_inference(CONFIG["pruned_ckpt"], onnx_path)
```

- [ ] **Step 3: 创建 inference/infer.py**

```python
# ============ inference/infer.py ============
"""单张/批量推理脚本"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet


def load_model(model_path=None, device=None):
    """
    加载训练好的模型

    Args:
        model_path: str — 权重文件路径
        device: str — 设备

    Returns:
        tuple: (model, device)
    """
    cfg = CONFIG
    model_path = model_path or cfg["pruned_ckpt"]
    device = device or cfg["device"]

    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, device


def get_transform(img_size=224):
    """获取推理用 transform"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def predict_image(image_path, model=None, device=None):
    """
    对单张图片进行预测

    Args:
        image_path: str — 图片路径
        model: nn.Module — 如为 None 则自动加载
        device: str

    Returns:
        dict: {class_name, confidence, top_k}
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    img = Image.open(image_path).convert('RGB')
    transform = get_transform(CONFIG["img_size"])
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)

    top_k = 3
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    return {
        "prediction": CONFIG["class_names"][top_indices[0][0].item()],
        "confidence": top_probs[0][0].item(),
        "top_k": [
            (CONFIG["class_names"][idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ],
    }


def predict_batch(image_paths, model=None, device=None):
    """
    批量预测

    Args:
        image_paths: list[str] — 图片路径列表
        model: nn.Module
        device: str

    Returns:
        list[dict]: 每张图片的预测结果
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    transform = get_transform(CONFIG["img_size"])
    results = []

    for path in image_paths:
        result = predict_image(path, model, device)
        result["path"] = path
        results.append(result)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top 3:")
        for name, prob in result['top_k']:
            print(f"  {name}: {prob:.4f}")
    else:
        print("Usage: python -m inference.infer <image_path>")
```

- [ ] **Step 4: 验证导入**

```bash
python -c "from inference.export_onnx import export_to_onnx, benchmark_inference; from inference.infer import predict_image, load_model; print('inference OK')"
```

- [ ] **Step 5: 提交**

```bash
git add inference/
git commit -m "feat: add ONNX export, benchmark, and inference modules"
```

---

### Task 11: Notebook 入口

**Files:**
- Modify: `coding_here.ipynb`

- [ ] **Step 1: 更新 coding_here.ipynb 为分阶段调用入口**

将 `coding_here.ipynb` 的内容替换为以下结构（每个 cell 对应一个独立运行阶段）：

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SkyEye — 天气分类\n",
    "## EfficientNet-B5 → 知识蒸馏 → B0 → 结构化剪枝 → ONNX 导出\n",
    "\n",
    "按顺序执行以下 Cell："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: 安装依赖\n",
    "!pip install torch torchvision timm onnx onnxruntime-gpu tqdm scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: 检查环境和配置\n",
    "from config import CONFIG\n",
    "import torch\n",
    "print(f'Device: {CONFIG[\"device\"]}')\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
    "print(f'Classes: {CONFIG[\"class_names\"]}')\n",
    "print(f'Teacher: {CONFIG[\"teacher_model\"]}, Student: {CONFIG[\"student_model\"]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: 数据准备（复制数据集到可写目录 + 验证）\n",
    "from data.dataset import prepare_data, create_dataloaders\n",
    "data_root = prepare_data()\n",
    "print(f'Data ready at: {data_root}')\n",
    "\n",
    "# 快速验证数据加载\n",
    "train_loader, val_loader, class_counts = create_dataloaders()\n",
    "print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: 训练教师模型 (EfficientNet-B5)\n",
    "from training.train_teacher import train_teacher\n",
    "teacher = train_teacher()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: 知识蒸馏 (B5 → B0)\n",
    "from training.distill_student import run_distillation\n",
    "student = run_distillation()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: 结构化剪枝 + 微调\n",
    "from training.prune_finetune import prune_and_finetune\n",
    "pruned_model = prune_and_finetune()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 7: ONNX 导出 + 推理测速\n",
    "from inference.export_onnx import export_to_onnx, benchmark_inference\n",
    "onnx_path = export_to_onnx('results/student_pruned_final.pth')\n",
    "benchmark_inference('results/student_pruned_final.pth', onnx_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 8 (可选): 单张图片推理测试\n",
    "from inference.infer import predict_image\n",
    "result = predict_image('test_image.jpg')  # 替换为实际图片路径\n",
    "print(f'Prediction: {result[\"prediction\"]} (Confidence: {result[\"confidence\"]:.4f})')\n",
    "for name, prob in result['top_k']:\n",
    "    print(f'  {name}: {prob:.4f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

- [ ] **Step 2: 提交**

```bash
git add coding_here.ipynb
git commit -m "feat: update notebook with staged training pipeline entry points"
```

---

## 执行顺序依赖

```
Task 1 (config) ← 所有模块依赖
    ↓
Task 2 (utils)  ← 独立
Task 3 (augmentations) ← 独立，Task 4 依赖
Task 4 (dataset) ← 依赖 Task 1, 3
Task 5 (model) ← 依赖 Task 1
    ↓
Task 6 (distill_wrapper) ← 依赖 Task 1, 5
Task 7 (train_teacher) ← 依赖 Task 1, 4, 5, 2
Task 8 (distill_student) ← 依赖 Task 1, 4, 5, 6
Task 9 (prune_finetune) ← 依赖 Task 1, 4, 5, 7
Task 10 (inference) ← 依赖 Task 1, 5
Task 11 (notebook) ← 依赖所有模块
```

并行组：
- Task 2, 3, 5 可并行执行（无相互依赖）
- Task 7, 8, 9, 10 可并行执行（仅依赖前面的基础模块）

---

## 验证清单

实现完成后，在 Mo 平台上执行以下验证：

1. `python -c "from config import CONFIG; print(CONFIG['class_names'])"` → 输出 6 个类别
2. `python -c "from data.dataset import create_dataloaders; train, val, _ = create_dataloaders()"` → 成功加载数据
3. `python -m training.train_teacher` → 训练 30 epochs，输出 `results/teacher_best.pth`
4. `python -m training.distill_student` → 蒸馏 40 epochs，输出 `results/student_distilled_best.pth`
5. `python -m training.prune_finetune` → 剪枝 3 轮 + 微调，输出 `results/student_pruned_final.pth`
6. `python -m inference.export_onnx` → 导出 `results/weather_model.onnx`，输出推理速度对比
