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

# 限制每个 worker 的 OpenMP/MKL 线程数，按 CPU 核心数自适应
# total = num_workers × OMP_NUM_THREADS，留一半给系统和 GPU 驱动
_omp_threads = str(min(8, max(2, (os.cpu_count() or 4) // 4)))
os.environ.setdefault("OMP_NUM_THREADS", _omp_threads)
os.environ.setdefault("MKL_NUM_THREADS", _omp_threads)

import torch

# GPU 性能优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---- 自适应硬件检测 ----

def _auto_num_workers():
    """自适应 DataLoader 线程数。

    Windows spawn 下保守取 2（避免启动开销过大），Linux 取 CPU 核数的一半，上限 4。
    注意：Windows 上必须通过独立 .py 脚本运行（local_train.py），且入口有
    if __name__ == "__main__" 守卫，否则 spawn 会导致递归创建进程死锁。
    """
    cpu_count = os.cpu_count() or 4
    if sys.platform == "win32":
        return min(2, max(0, cpu_count // 2))
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
    "writable_root": "_data/weather",  # 将只读数据集合并复制到此可写目录
    # 类名别名表：自动发现时，将不同命名的类目录映射到标准 class_names（形容词）
    "class_aliases": {
        # ---- foggy 的别名（雾 / 能见度降低类） ----
        "haze": "foggy",
        "fog": "foggy",
        "fogsmog": "foggy",       # 雾霾（fog + smog 连写）
        "smog": "foggy",          # 雾霾
        # ---- snowy 的别名（冰雪 / 冻结降水类） ----
        "snow": "snowy",
        # ---- rainy 的别名 ----
        "rain": "rainy",
        # ---- thundery 的别名（雷暴 / 强对流类） ----
        "thunder": "thundery",
        "thunderstorm": "thundery",
        "lightning": "thundery",  # 闪电
    },
    "class_names": ["cloudy", "foggy", "rainy", "snowy", "sunny", "thundery", "other"],
    "skip_classes": ["other"],
    "img_size": 380,               # EfficientNet-B4 原生分辨率
    "batch_size": 8,              # 手动指定，8GB 显存 + B4@380 的稳妥值
    "val_split": 0.15,             # 验证集比例

    # ---- 教师模型 ----
    "teacher_model": "efficientnet_b4",  # timm 模型名
    "teacher_pretrained": True,
    "teacher_epochs": 15,            # 全流程总轮数（P1: 9 + P2: 6）
    "teacher_phase1_epochs": 9,     # Phase 1: 标准采样 + MixUp
    "teacher_phase2_epochs": 6,     # Phase 2: DRW 过采样 + MixUp
    "teacher_lr": 5e-5,  # 380 原生分辨率下微调，保守 LR 保护预训练特征
    "teacher_weight_decay": 1e-4,
    "warmup_epochs": 2,             # 学习率 warmup 轮数（LinearLR 0.1→1.0）

    # ---- 知识蒸馏 ----
    "student_model": "efficientnet_b0",  # timm 模型名
    "kd_temperature": 4.0,               # 蒸馏温度 T
    "kd_alpha": 0.7,                     # 软标签损失权重
    "kd_feature_weight": 0.1,            # 中间层特征损失权重
    "kd_epochs": 15,                     # 知识蒸馏轮数
    "kd_lr": 1e-3,

    # ---- 结构化剪枝（可选，仅速度需要时启用）----
    "prune_ratio": 0.4,           # 最终剪枝比例
    "prune_iterations": 2,        # 渐进剪枝轮数（2轮: 20%→40%）
    "prune_finetune_epochs": 5,   # 剪枝后微调轮数
    "prune_finetune_lr": 1e-4,

    # ---- 通用 ----
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "fp16": True,                 # 混合精度训练（仅 CUDA 生效）
    "use_tb": True,              # TensorBoard 日志（需 pip install tensorboard）
    "num_workers": _auto_num_workers(),  # 自适应：Win→2, Linux→min(4, cpu//2)
    "scheduler": "cosine",        # cosine / plateau
    "label_smoothing": 0.0,  # 关闭：FocalLoss 自带 max-entropy 正则化 (NeurIPS 2020)
    # 且 LS 与 FL 梯度机制冲突 (NeurIPS 2021) + LS 教师损害 KD 软标签 (Müller, NeurIPS 2019)
    # Per-class label smoothing（方案 D）：对易混淆类用更高平滑值
    "per_class_label_smoothing": {
        "sunny": 0.1, "cloudy": 0.1,
    },
    "use_focal_loss": True,       # 处理类别不平衡
    "focal_gamma": 1.0,            # 降为 1，让困难样本（cloudy）拿到梯度
    "mixup_alpha": 0.2,            # MixUp 混合强度 (Zhang et al., ICLR 2018)；0.0=关闭
    "ema_decay": 0.99997,          # EMA 衰减，平滑窗口 ~33k steps ≈ 7 epochs

    # ---- Logit Adjustment（方案 A）----
    # 推理时 logit bias：降低 sunny 门槛，提高 cloudy 门槛
    # bias > 0 → 更难被判为该类；bias < 0 → 更容易被判为该类
    "logit_bias": {
        "sunny": -0.5,
        "cloudy": 0.3,
    },

    # ---- Cost-Sensitive Loss（方案 B）----
    # 对特定混淆方向施加额外惩罚
    "confusion_penalty_weight": 0.3,  # sunny→cloudy 额外惩罚强度

    # ---- 推理 ----
    "inference_device": "cpu",           # 比赛评测用 CPU 推理
    "use_int8_quantization": True,       # CPU 推理时启用 INT8 量化加速

    # ---- 路径 ----
    "teacher_ckpt_dir": "results/checkpoints/teacher",
    "teacher_ckpt": "results/teacher_best.pth",
    "distill_ckpt_dir": "results/checkpoints/distill",
    "distilled_ckpt": "results/student_distilled_best.pth",
    "pruned_ckpt": "results/student_pruned_final.pth",
    "onnx_path": "results/weather_model.onnx",
    "onnx_int8_path": "results/weather_model_int8.onnx",
}

CONFIG["num_classes"] = len(CONFIG["class_names"]) - len(CONFIG["skip_classes"])
