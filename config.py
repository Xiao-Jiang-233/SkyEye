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

# 限制每个 worker 的线程数，防止 DataLoader 多进程时 CPU 线程爆炸
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

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
        "frost": "snowy",         # 霜冻 → 冰晶
        "glaze": "snowy",         # 雨凇 → 冻雨冰壳
        # ---- rainy 的别名 ----
        "rain": "rainy",
        # ---- thundery 的别名（雷暴 / 强对流类） ----
        "thunder": "thundery",
        "thunderstorm": "thundery",
        "lightning": "thundery",  # 闪电
        "hail": "thundery",       # 冰雹 → 强对流产物
        # ---- sunny 的别名（晴 / 好天气类） ----
        "rainbow": "sunny",       # 彩虹 → 需阳光，雨后晴天现象
    },
    "num_classes": 6,  # 当前仅训练 6 类（dew/rime/sandstorm 暂缓）
    "class_names": ["cloudy", "dew", "foggy", "rainy", "rime", "sandstorm", "snowy", "sunny", "thundery"],
    # 暂时跳过的类：主数据集（weather_classification）中没有这些类型，训练时忽略
    # 当主数据集扩展后，从此列表中移除即可启用
    "skip_classes": ["dew", "rime", "sandstorm"],
    "img_size": 380,               # EfficientNet-B4 原生分辨率
    "batch_size": 8,              # 手动指定，8GB 显存 + B4@380 的稳妥值
    "val_split": 0.15,             # 验证集比例

    # ---- 教师模型 ----
    "teacher_model": "efficientnet_b4",  # timm 模型名
    "teacher_pretrained": True,
    "teacher_epochs": 15,            # 15轮给SAM后5轮充分收敛
    "teacher_lr": 5e-5,  # 380 原生分辨率下微调，保守 LR 保护预训练特征
    "teacher_weight_decay": 1e-4,
    "warmup_epochs": 2,             # 学习率 warmup 轮数（LinearLR 0.1→1.0）

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
    "use_tb": True,              # TensorBoard 日志（需 pip install tensorboard）
    "num_workers": _auto_num_workers(),  # 自适应：Win→0, Linux→min(4, cpu//2)
    "scheduler": "cosine",        # cosine / plateau
    "label_smoothing": 0.0,  # 关闭：FocalLoss 自带 max-entropy 正则化 (NeurIPS 2020)
    # 且 LS 与 FL 梯度机制冲突 (NeurIPS 2021) + LS 教师损害 KD 软标签 (Müller, NeurIPS 2019)
    "use_focal_loss": True,       # 处理类别不平衡
    "focal_gamma": 1.0,            # 降为 1，让困难样本（cloudy）拿到梯度
    "mixup_alpha": 0.2,            # MixUp 混合强度 (Zhang et al., ICLR 2018)；0.0=关闭
    "sam_rho": 0.05,               # SAM 优化器扰动半径（平坦极小值 → 泛化好）
    "ema_decay": 0.99997,          # EMA 衰减，平滑窗口 ~33k steps ≈ 7 epochs

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
