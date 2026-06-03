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
    "teacher_model": "efficientnet_b5",  # timm 模型名
    "teacher_pretrained": True,
    "teacher_epochs": 30,
    "teacher_lr": 1e-3,
    "teacher_weight_decay": 1e-4,

    # ---- 知识蒸馏 ----
    "student_model": "efficientnet_b0",  # timm 模型名
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
