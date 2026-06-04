# SkyEye — 天气图片分类

基于 **EfficientNet-B5 → 知识蒸馏 → B0 → 结构化剪枝 → ONNX → INT8 量化** 的六类天气分类管线。

## 功能

对输入天气图片进行六分类预测：

| 类名 | 中文 |
|------|------|
| `cloudy` | 多云 |
| `foggy` | 雾霾 |
| `rainy` | 雨天 |
| `snowy` | 雪天 |
| `sunny` | 晴天 |
| `thundery` | 雷暴 |

## 技术方案

```text
EfficientNet-B5 (Teacher)
    ↓ Knowledge Distillation (软标签 + 特征对齐)
EfficientNet-B0 (Student)
    ↓ Structured Pruning (渐进 2 轮: 20% → 40%)
Pruned EfficientNet-B0
    ↓ ONNX Export (FP32)
    ↓ INT8 Dynamic Quantization
CPU Inference (ONNX Runtime)
```

训练 GPU，推理 CPU。总时限 70 分钟。

## 运行环境

| 组件 | 版本 |
| --- | --- |
| **Python** | 3.13.13 |
| **PyTorch** | 2.8.0+cu128 |
| **torchvision** | 0.23.0 |
| **timm** | 1.0.27 |
| **onnx** | 1.21.0 |
| **onnxruntime-gpu** | 1.26.0 |
| **平台** | [Mo Platform](https://momodel.cn) (JupyterLab + GPU + CPU 推理) |

> 预训练模型下载已配置 HF 镜像 (`hf-mirror.com`)，国内可正常访问。

## 项目结构

```text
SkyEye/
├── main.ipynb                     # Notebook 入口，按顺序执行训练管线
├── prepare_datasets.ipynb         # 数据集准备（备用）
├── config.py                      # 超参数统一管理 + HF 镜像配置
├── data/
│   ├── augmentations.py           # Train/Val 增强策略 (RandAugment)
│   └── dataset.py                 # 多源合并 + ImageFolder + 类别权重
├── models/
│   ├── weather_efficientnet.py    # EfficientNet 封装 + 中间层 hook
│   └── distill_wrapper.py         # 软标签 + 特征蒸馏训练器
├── training/
│   ├── train_teacher.py           # 教师训练 (FocalLoss + 混合精度)
│   ├── distill_student.py         # 知识蒸馏入口
│   └── prune_finetune.py          # 结构化剪枝 + 渐进微调
├── inference/
│   ├── export_onnx.py             # ONNX 导出 + INT8 量化 + CPU 测速
│   └── infer.py                   # 单张/批量推理
└── utils/
    ├── metrics.py                 # F1 / 混淆矩阵 / 分类报告
    └── logger.py                  # TensorBoard + Mo 平台 JSON 日志
```

## 数据集

6 类天气图片 × 各 10,000 张 = 共 **60,000** 张。

数据源合并到 `_data/weather/`（不入 git），支持多源自动合并 + 类名映射。

> Mo 平台不允许 `.` 开头的文件/目录，故使用 `_data/` 前缀。

## 训练流程（70 分钟 GPU 时限）

| 阶段 | 内容 | 预估耗时 |
|------|------|----------|
| 1. Train Teacher | EfficientNet-B5, 10 epochs, FocalLoss | ~30 min |
| 2. Knowledge Distillation | B5 → B0, 15 epochs, T=4, α=0.7 | ~15 min |
| 3. Structured Pruning | 渐进 2 轮 (20%→40%) + Fine-tune 5 epoch × 2 | ~5 min |
| 4. ONNX Export + INT8 | FP32 → ONNX → INT8 动态量化 | ~3 min |
| 5. CPU Inference | ONNX Runtime CPUExecutionProvider | <100ms/img |

## 依赖安装

```bash
# PyTorch + torchvision（CUDA 12.6）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
pip install timm==1.0.27 onnx==1.21.0 onnxruntime-gpu==1.26.0 tqdm scikit-learn
```

## 相关文档

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 项目开发指南 |
| [docs/接口文档.md](docs/接口文档.md) | 模块 API 接口文档 |

| [设计文档](docs/superpowers/specs/) | 技术方案设计 |
