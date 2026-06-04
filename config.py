# ============ config.py ============
"""
SkyEye 天气分类项目 — 超参数配置中心
所有模块通过 `from config import CONFIG` 统一获取参数
"""
import os
import sys

# 使用国内 HF 镜像下载 timm 预训练权重（hf-mirror.com）
# 必须在导入 timm / huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

# GPU 性能优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---- 自适应硬件检测 ----

def _auto_batch_size():
    """根据 GPU 显存自适应调整 batch_size，CPU 回退到 8。"""
    if not torch.cuda.is_available():
        return 8
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 16:
        return 32
    elif total_gb >= 12:
        return 24
    elif total_gb >= 8:
        return 16
    elif total_gb >= 6:
        return 8
    elif total_gb >= 4:
        return 4
    else:
        return 2


def _auto_num_workers():
    """自适应 DataLoader 线程数。

    Windows spawn 机制下多线程会卡死，强制返回 0。
    Linux 下取 CPU 核数的一半，上限 4（避免 Docker 共享内存溢出）。
    """
    if sys.platform == "win32":
        return 0
    cpu_count = os.cpu_count() or 4
    return min(4, max(0, cpu_count // 2))

CONFIG = {
    # ---- 数据 ----
    # 多数据集自动处理：
    #   "auto" — 自动扫描 datasets/ 下所有导入，发现 weather_classification/ 或 .zip，
    #            内置别名表自动映射类名差异，合并到 writable_root
    #   列表 — 手动指定，每个条目可以是:
    #           1. 字符串路径（类名需与 class_names 一致）
    #           2. dict {path, class_map}（手动映射：foggy→haze, snowy→snow）
    #   支持 .zip 文件（自动解压到临时目录）
    "data_roots": "auto",
    "writable_root": "_data/weather",  # 将只读数据集合并复制到此可写目录（Mo 平台不允许 . 开头的文件/目录）
    # 类名别名表：自动发现时，将不同命名的类目录映射到标准 class_names（形容词）
    "class_aliases": {
        # foggy 的别名
        "haze": "foggy",
        "fog": "foggy",
        # snowy 的别名
        "snow": "snowy",
        # rainy 的别名
        "rain": "rainy",
        # thundery 的别名
        "thunder": "thundery",
        "thunderstorm": "thundery",
        "lightning": "thundery",
    },
    "num_classes": 6,
    "class_names": ["cloudy", "foggy", "rainy", "snowy", "sunny", "thundery"],
    "img_size": 224,               # EfficientNet 标准输入
    "batch_size": _auto_batch_size(),  # 根据 GPU 显存自适应
    "val_split": 0.15,             # 验证集比例

    # ---- 教师模型 ----
    "teacher_model": "efficientnet_b5",  # timm 模型名
    "teacher_pretrained": True,
    "teacher_epochs": 10,            # 70分钟时限下缩减至10轮
    "teacher_lr": 1e-3,
    "teacher_weight_decay": 1e-4,

    # ---- 知识蒸馏 ----
    "student_model": "efficientnet_b0",  # timm 模型名
    "kd_temperature": 4.0,               # 蒸馏温度 T
    "kd_alpha": 0.7,                     # 软标签损失权重
    "kd_feature_weight": 0.1,            # 中间层特征损失权重
    "kd_epochs": 15,                     # 70分钟时限下缩减至15轮
    "kd_lr": 1e-3,

    # ---- 结构化剪枝 ----
    "prune_ratio": 0.4,           # 最终剪枝比例
    "prune_iterations": 2,        # 渐进剪枝轮数（2轮: 20%→40%）
    "prune_finetune_epochs": 5,   # 剪枝后微调轮数（70分钟内缩减）
    "prune_finetune_lr": 1e-4,

    # ---- 通用 ----
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "fp16": True,                 # 混合精度训练（仅 CUDA 生效）
    "num_workers": _auto_num_workers(),  # 自适应：Win→0, Linux→min(4, cpu//2)
    "scheduler": "cosine",        # cosine / plateau
    "label_smoothing": 0.1,
    "use_focal_loss": True,       # 处理类别不平衡
    "focal_gamma": 2.0,

    # ---- 推理 ----
    "inference_device": "cpu",           # 比赛评测用 CPU 推理
    "use_int8_quantization": True,       # CPU 推理时启用 INT8 量化加速

    # ---- 路径 ----
    "teacher_ckpt": "results/teacher_best.pth",
    "distilled_ckpt": "results/student_distilled_best.pth",
    "pruned_ckpt": "results/student_pruned_final.pth",
    "onnx_path": "results/weather_model.onnx",
    "onnx_int8_path": "results/weather_model_int8.onnx",
}
