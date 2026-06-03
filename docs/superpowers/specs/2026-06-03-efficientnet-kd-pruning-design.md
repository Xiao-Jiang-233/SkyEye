# EfficientNet + 知识蒸馏 + 结构化剪枝 — 设计文档

> 日期：2026-06-03 | 项目：SkyEye | 赛题：智海算法调优

## 一、项目目标与约束

针对六类天气图片分类任务，通过 **EfficientNet-B5→B0 知识蒸馏 + 结构化剪枝 + ONNX 导出 + INT8 量化** 的组合方案：
- **精度**：Macro F1 ≥ 0.84
- **速度**：CPU 推理 ≤ 3ms/张（ONNX INT8）
- **时限**：GPU 训练 + CPU 推理，总时限 70 分钟
- **参数**：最终模型 ≤ 3M

## 二、与项目实际的适配

| 适配项 | 原方案 | 实际调整 |
|--------|--------|----------|
| 类别名 | `snowy` | `snow`（与数据集目录名一致） |
| 数据路径 | `./data/weather` | `datasets/<hash>/weather_classification/` → 需复制到可写目录 |
| 结果路径 | 无指定 | `results/`（Mo 平台约定） |
| 平台限制 | 通用 | datasets/ 只读、Mo 平台内存/显存限制 |
| 开发方式 | 纯 .py 模块 | `.py` 模块 + `coding_here.ipynb` 入口调用 |

## 三、项目结构

```
SkyEye/
├── coding_here.ipynb              # Notebook 入口（分阶段调用）
├── config.py                      # 超参数统一管理
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # ImageFolder 加载 + 类别权重计算
│   └── augmentations.py           # train/val 增强策略（含 CutMix/MixUp 可选）
├── models/
│   ├── __init__.py
│   ├── weather_efficientnet.py    # EfficientNet 封装 + 中间层 hook
│   └── distill_wrapper.py         # 蒸馏损失 + 特征投影 + 训练循环
├── training/
│   ├── __init__.py
│   ├── train_teacher.py           # 教师模型训练（FocalLoss + 混合精度）
│   ├── distill_student.py         # 知识蒸馏入口（加载教师 → 蒸馏学生）
│   └── prune_finetune.py          # 结构化剪枝 + 渐进微调
├── inference/
│   ├── __init__.py
│   ├── export_onnx.py             # ONNX 导出 + 简化 + 推理测速
│   └── infer.py                   # 单张/批量推理脚本
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # F1 / 混淆矩阵 / classification_report
│   └── logger.py                  # 训练日志 + TensorBoard
├── results/                       # 模型权重和训练结果
│   ├── teacher_best.pth
│   ├── student_distilled_best.pth
│   └── student_pruned_final.pth
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-03-efficientnet-kd-pruning-design.md
```

## 四、数据流水线

### 4.1 数据加载

```
数据源 (datasets/.../weather_classification/, 只读)
    │
    ├── cloudy/  (10,000)
    ├── haze/    (10,000)
    ├── rainy/   (10,000)
    ├── snow/    (10,000)
    ├── sunny/   (10,000)
    └── thunder/ (10,000)
    │
    ▼ 复制到 .data/weather/ (可写)
    │
    ▼ ImageFolder → TrainLoader / ValLoader
```

### 4.2 数据增强

| 阶段 | 策略 |
|------|------|
| Train | RandomResizedCrop(224) → RandomHorizontalFlip → RandAugment(N=2,M=9) → Normalize |
| Val | Resize(256) → CenterCrop(224) → Normalize |
| 可选 | CutMix(α=1.0) / MixUp(α=0.2) 处理类别不平衡 |

### 4.3 类别权重

通过统计各类别样本数自动计算 FocalLoss 的 α 参数，缓解 thunder/haze 等潜在的长尾效应。

## 五、模型架构

### 5.1 Teacher: EfficientNet-B5

```
EfficientNet-B5 (timm, pretrained ImageNet)
    ├── backbone (frozen early layers)
    ├── 中间层 hook → 捕获 stage 特征用于 KD
    └── 分类头: Dropout → Linear(2048→512) → SiLU → Dropout → Linear(512→6)
```

### 5.2 Student: EfficientNet-B0

```
EfficientNet-B0 (timm, pretrained ImageNet)
    ├── backbone
    ├── 中间层 hook → 与 Teacher 对应 stage 对齐
    └── 分类头: Dropout → Linear(1280→512) → SiLU → Dropout → Linear(512→6)
```

### 5.3 知识蒸馏

```
                         ┌──────────────────┐
                         │ Teacher (frozen) │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       soft_labels            stage_feats          (无)
              │                   │
              │    ┌──────────────┘
              │    │
              ▼    ▼
    Loss = α × KL(soft_S, soft_T) × T²
         + (1-α) × CE(student, labels)
         + β × MSE(proj(feat_S), feat_T)
```

- T=4, α=0.7, β=0.1
- 特征对齐使用 1×1 卷积投影层，取最后 N 个 stage

## 六、训练流程

### 6.1 三阶段流水线（70 分钟 GPU 时限适配）

```
Step 1: Train Teacher (10 epochs)                    ~30 min
    ├── 损失: FocalLoss (γ=2.0, 动态 α)
    ├── 优化器: AdamW (lr=1e-3, wd=1e-4)
    ├── 调度器: CosineAnnealingLR
    ├── 混合精度: GradScaler (fp16)
    └── 输出: teacher_best.pth (按 F1 择优)

Step 2: Knowledge Distillation (15 epochs)            ~15 min
    ├── 教师: 加载 teacher_best.pth, 冻结
    ├── 学生: EfficientNet-B0, ImageNet 预训练
    ├── 损失: α×KL + (1-α)×CE + β×FeatureMSE
    ├── 优化器: AdamW (lr=1e-3, wd=1e-4)
    └── 输出: student_distilled_best.pth

Step 3: Structured Pruning + Fine-tune                ~5 min
    ├── 策略: 渐进式 2 轮 (20% → 40%)
    ├── 方法: L2-norm 结构化通道剪枝 (只剪 1×1 conv)
    ├── 每轮后微调 5 epochs (lr=1e-4)
    ├── 固化 mask → 生成密集小模型
    └── 输出: student_pruned_final.pth

Step 4: ONNX Export + INT8 Quantization               ~3 min
    └── 输出: weather_model.onnx, weather_model_int8.onnx
```

### 6.2 性能优化

- **混合精度** (fp16)：训练加速 40%+，降低显存占用
- **梯度累积**：若 OOM，batch_size 减半 + gradient_accumulation_steps=2
- **num_workers=4**：数据加载并行

## 七、推理与导出

### 7.1 ONNX 导出

```
PyTorch Model (.pth)
    → torch.onnx.export (opset=13, constant folding, CPU 导出)
    → weather_model.onnx (FP32)
```

### 7.2 INT8 动态量化

```
weather_model.onnx (FP32)
    → onnxruntime.quantization.quantize_dynamic (QInt8)
    → weather_model_int8.onnx
    → 体积 ~50%，CPU 推理加速 2-4×
```

### 7.3 CPU 推理测速

比赛评测使用 CPU。对比 ONNX FP32 vs INT8 延迟，均在 `CPUExecutionProvider` 下测量。

## 八、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 224 | EfficientNet 标准输入 |
| `batch_size` | 32 | 可根据显存调整 |
| `kd_temperature` | 4.0 | 软标签平滑度 |
| `kd_alpha` | 0.7 | 软标签:硬标签 权重比 |
| `kd_feature_weight` | 0.1 | 特征对齐损失权重 |
| `prune_ratio` | 0.4 | 最终剪枝比例 |
| `prune_iterations` | 2 | 渐进剪枝轮数（70min 时限） |
| `focal_gamma` | 2.0 | FocalLoss 聚焦参数 |
| `label_smoothing` | 0.1 | 用于蒸馏的硬标签 |
| `seed` | 42 | 全局随机种子 |
| `fp16` | True | 混合精度训练 |

## 九、预期效果（CPU 推理）

| 阶段 | 模型 | 参数量 | CPU 推理 | Macro F1 |
|------|------|--------|----------|----------|
| ① Teacher | EfficientNet-B5 | 30.4M | — | ~0.88 |
| ② 直接训练 | EfficientNet-B0 | 5.3M | ~15ms | ~0.81 |
| ③ KD 蒸馏 | B0 (distilled) | 5.3M | ~12ms | ~0.85 |
| ④ KD+剪枝 | B0 (pruned 40%) | ~2.8M | ~6ms | ~0.84 |
| ⑤ ONNX FP32 | ONNX Runtime | ~2.8M | ~4ms | ~0.84 |
| ⑥ ONNX INT8 | ONNX Runtime INT8 | ~1.5M | **~2ms** | ~0.84 |

## 十、可选扩展

| 扩展 | 说明 | 优先级 |
|------|------|--------|
| CutMix/MixUp 数据增强 | 处理类别不平衡，提升泛化 | 中 |
| 5-Fold Cross Validation | 更高精度，但训练时间 ×5 | 低 |
| TensorRT 部署 | GPU 推理极致加速 | 低 |
| app_spec.yml | Mo 平台模型部署接口定义 | 后续 |
