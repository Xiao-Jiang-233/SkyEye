# SkyEye — 天气图片分类

基于 **EfficientNet-B4 → 知识蒸馏 → B0 → ONNX → INT8 量化** 的天气分类管线。

## 功能

对输入天气图片进行分类预测（当前 6 类 + 1 兜底类）：

| 类名       | 中文                       | 训练状态   |
| ---------- | -------------------------- | ---------- |
| `cloudy`   | 多云                       | ✓          |
| `foggy`    | 雾霾                       | ✓          |
| `rainy`    | 雨天                       | ✓          |
| `snowy`    | 雪天                       | ✓          |
| `sunny`    | 晴天                       | ✓          |
| `thundery` | 雷暴                       | ✓          |
| `other`    | 其他（dew/rime/sandstorm） | ⏭ 暂不训练 |

## 技术方案

```text
EfficientNet-B4 (Teacher)
    ↓ Knowledge Distillation (软标签 + 特征对齐)
EfficientNet-B0 (Student)
    ↓ ONNX Export (FP32)
    ↓ INT8 Dynamic Quantization
CPU Inference (ONNX Runtime)
```

训练本地 GPU 不限时，推理 CPU 总时限 70 分钟。

## 运行环境

| 组件            | 版本                                                              |
| --------------- | ----------------------------------------------------------------- |
| **Python**      | 3.13.13                                                           |
| **PyTorch**     | 2.12.0                                                            |
| **torchvision** | 0.27.0                                                            |
| **timm**        | 1.0.27                                                            |
| **onnx**        | 1.21.0                                                            |
| **onnxruntime** | 1.26.0                                                            |
| **平台**        | Windows 11 + AMD Ryzen 9 9955HX + RTX 5070 (Blackwell, CUDA 13.0) |

> 预训练模型下载已配置 HF 镜像 (`hf-mirror.com`)，国内可正常访问。

## 项目结构

```text
SkyEye/
├── main.ipynb                     # Jupyter Notebook 入口，按顺序执行训练管线
├── scripts/
│   └── eval_full.py               # 全量数据集评估
├── config.py                      # 超参数统一管理 + HF 镜像配置
├── data/
│   ├── augmentations.py           # Train/Val 增强策略 (RandAugment)
│   └── dataset.py                 # 多源合并 + ImageFolder + 类别权重
├── models/
│   ├── weather_efficientnet.py    # EfficientNet 封装 + 中间层 hook
│   └── distill_wrapper.py         # 软标签 + 特征蒸馏训练器
├── training/
│   ├── train_teacher.py           # 教师训练 (FocalLoss + MixUp + EMA + DRW + ConfusionPenalty + LogitAdj)
│   └── distill_student.py         # 知识蒸馏入口
├── inference/
│   ├── export_onnx.py             # ONNX 导出 + INT8 量化 + CPU 测速
│   └── infer.py                   # 单张/批量推理
└── utils/
    ├── metrics.py                 # F1 / 混淆矩阵 / 分类报告
    └── logger.py                  # TensorBoard 日志
```

## 数据集

主数据集 6 类 × 各 10,000 张 = **60,000** 张 + 补充数据集 6,862 张，共 **66,862** 张。dew/rime/sandstorm 通过 `class_aliases` 映射到 `other` 兜底类（当前不训练）。

数据源合并到 `_data/weather/`（不入 git），支持多源自动合并 + 类名映射。

> 数据集目录使用 `_data/` 前缀，已在 `.gitignore` 中排除。

## 训练流程（本地不限时）

| 阶段                      | 内容                                        | 预估耗时   |
| ------------------------- | ------------------------------------------- | ---------- |
| 1. Train Teacher          | EfficientNet-B4, 15 epochs (P1 9 + P2 6)   | ~3h 40min  |
| 2. Knowledge Distillation | B4 → B0, 15 epochs, T=4, α=0.7              | ~15 min    |
| 3. ONNX Export + INT8     | FP32 → ONNX → INT8 动态量化                 | ~3 min     |
| 4. CPU Inference          | ONNX Runtime CPUExecutionProvider           | <100ms/img |

### 训练监控（TensorBoard）

每个阶段自动写入 `results/tb_results/`，训练中/结束后均可查看：

```bash
tensorboard --logdir results/tb_results/
# 浏览器打开 http://localhost:6006
```

SCALARS 页可对比各阶段的 loss / F1 / Accuracy / per-class F1 曲线。

## 依赖安装

```bash
pip install -r requirements.txt
```

Jupyter Notebook 中按顺序执行各 Cell 即可完成训练管线。

## 相关文档

| 文档 | 说明 |
| ---- | ---- |
| [CLAUDE.md](CLAUDE.md) | 项目开发指南（配置、数据集、训练策略） |
| [docs/接口文档.md](docs/接口文档.md) | 模块 API 接口文档 |
| [docs/design-efficientnet-kd-pruning.md](docs/design-efficientnet-kd-pruning.md) | 技术方案设计 |
| [docs/plan-efficientnet-kd-pruning.md](docs/plan-efficientnet-kd-pruning.md) | 实施计划 |
| [docs/competition-rules.md](docs/competition-rules.md) | 比赛规则 |
| [docs/competition-faq.md](docs/competition-faq.md) | 比赛 FAQ |
