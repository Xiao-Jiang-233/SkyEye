# SkyEye — 天气图片分类

基于 **EfficientNet-B5 → 知识蒸馏 → B0 → 结构化剪枝 → ONNX** 的六类天气分类管线。

## 功能

对输入天气图片进行六分类预测：cloudy（多云）、haze（雾霾）、rainy（雨天）、snow（雪天）、sunny（晴天）、thunder（雷暴）。

## 技术方案

```text
EfficientNet-B5 (Teacher) → Knowledge Distillation → EfficientNet-B0 (Student) → Structured Pruning → ONNX Export
```

## 运行环境

| 组件 | 版本 |
| --- | --- |
| **Python** | 3.9.5 |
| **PyTorch** | 2.3.1 |
| **torchvision** | 0.18.1 |
| **timm** | 1.0.8 |
| **onnx** | 1.16.1 |
| **onnxruntime-gpu** | 1.18.1 |
| **平台** | [Mo Platform](https://momodel.cn) (JupyterLab + GPU) |

## 项目结构

```text
SkyEye/
├── coding_here.ipynb              # Notebook 入口，分阶段调用
├── config.py                      # 超参数统一管理
├── data/
│   ├── augmentations.py           # Train/Val 增强策略
│   └── dataset.py                 # ImageFolder 加载 + 类别权重
├── models/
│   ├── weather_efficientnet.py    # EfficientNet 封装 + 中间层 hook
│   └── distill_wrapper.py         # 软标签 + 特征蒸馏训练器
├── training/
│   ├── train_teacher.py           # 教师训练 (FocalLoss + 混合精度)
│   ├── distill_student.py         # 知识蒸馏入口
│   └── prune_finetune.py          # 结构化剪枝 + 渐进微调
├── inference/
│   ├── export_onnx.py             # ONNX 导出 + 测速
│   └── infer.py                   # 单张/批量推理
└── utils/
    ├── metrics.py                 # F1 / 混淆矩阵 / 分类报告
    └── logger.py                  # TensorBoard 日志
```

## 数据集

6 类天气图片 × 各 10,000 张 = 共 **60,000** 张。

| 类别 | 数量 |
| --- | --- |
| cloudy（多云） | 10,000 |
| haze（雾霾） | 10,000 |
| rainy（雨天） | 10,000 |
| snow（雪天） | 10,000 |
| sunny（晴天） | 10,000 |
| thunder（雷暴） | 10,000 |

## 训练流程

1. **Train Teacher:** EfficientNet-B5, 30 epochs, FocalLoss
2. **Knowledge Distillation:** B5 → B0, 40 epochs, T=4, α=0.7
3. **Structured Pruning:** 渐进 3 轮 (15% → 27.5% → 40%) + Fine-tune
4. **ONNX Export:** ONNX Runtime 推理

## 依赖安装

```bash
pip install -r requirements.txt
```
