# EfficientNet + 知识蒸馏 + 结构化剪枝 — 设计文档

> 日期：2026-06-06 | 项目：SkyEye | 赛题：智海算法调优 | 状态：教师训练完成

## 一、项目目标与约束

针对天气图片分类任务，通过 **EfficientNet-B4→B0 知识蒸馏 + 结构化剪枝 + ONNX 导出 + INT8 量化** 的组合方案：
- **精度**：Macro F1 ≥ 0.87
- **速度**：CPU 推理 ≤ 3ms/张（ONNX INT8）
- **时限**：GPU 训练 + CPU 推理，总时限 70 分钟
- **参数**：最终模型 ≤ 3M
- **当前教师最优**：Macro F1 0.8933 / Acc 89.16%（全量 60k 评估）

## 二、类别架构

`class_names` 共 7 类（6 核心 + `other` 兜底），当前训练 6 类：

| 类 | 训练 | 数据量 | 说明 |
|---|---|---|---|
| cloudy | ✓ | 10,000 | 多云 — 最大混淆源 |
| foggy | ✓ | 10,000 | 雾霾 |
| rainy | ✓ | 10,000 | 雨天 |
| snowy | ✓ | 10,000 | 雪天 — F1 > 0.95 |
| sunny | ✓ | 10,000 | 晴天 — 1510 张误判为 cloudy |
| thundery | ✓ | 10,000 | 雷暴 — F1 > 0.97 |
| other | ⏭ skip | ~2,550 | dew/rime/sandstorm → 归入 other |

## 三、与项目实际的适配

| 适配项 | 原方案 | 实际调整 |
|--------|--------|----------|
| 教师模型 | EfficientNet-B5 | **B4**（380px 原生分辨率，8GB 显存可行） |
| 输入尺寸 | 224 | **380**（B4 原生分辨率，不放缩） |
| 类别架构 | 9 类 | **7 类**（6 + other 兜底），dew/rime/sandstorm → other |
| Batch Size | 32 | **8**（380px + B4，显存限制） |
| 混合精度 | fp16 + GradScaler | **BF16 autocast**（RTX 5070 原生，无需 Scaler） |
| Label Smoothing | 0.1 | **0.0**（保护 KD 软标签质量） |
| 增强 | RandAugment | + **MixUp α=0.2** + cloudy **DRW 过采样 2×** |

## 四、项目结构

```
SkyEye/
├── main.ipynb                      # Jupyter 入口
├── config.py                       # 超参数 + HF 镜像
├── scripts/
│   ├── local_train.py              # CLI 训练入口
│   └── eval_full.py                # 全量 60k 评估
├── data/
│   ├── dataset.py                  # 多源合并 + DataLoader
│   └── augmentations.py            # RandAugment + MixUp
├── models/
│   ├── weather_efficientnet.py     # EfficientNet 封装
│   └── distill_wrapper.py          # 蒸馏训练器
├── training/
│   ├── train_teacher.py            # 教师训练（FocalLoss+DRW+SAM+EMA）
│   ├── distill_student.py          # 知识蒸馏
│   └── prune_finetune.py           # 剪枝 + 微调
├── inference/
│   ├── export_onnx.py              # ONNX + INT8 + 测速
│   └── infer.py                    # 单张/批量推理
├── utils/
│   ├── metrics.py
│   └── logger.py                   # TensorBoard SCALARS
└── results/
    ├── teacher_best.pth
    ├── checkpoints/                # 每 epoch 周期备份 × 20
    └── tb_results/
```

## 五、数据流水线

### 5.1 数据加载

```
datasets/<hash>/weather_classification/ (只读, 60k)
    + datasets/<hash>/weather-dataset.zip (补充 6.8k)
        │
        ▼ class_aliases: haze→foggy, snow→snowy, dew→other, ...
        │
        ▼ 复制到 _data/weather/（可写，仅首次）
        │
        ▼ ImageFolder → TrainLoader / ValLoader (stratified 85/15)
```

### 5.2 数据增强

| 阶段 | 策略 |
|------|------|
| Train | RandomResizedCrop(380) → RandAugment → MixUp(α=0.2, Fast 阶段) → Normalize |
| Val | Resize → CenterCrop(380) → Normalize |
| DRW | 前 60% epoch 标准采样 → 后 40% cloudy 过采样 2× |
| SAM | 后 5 epoch，**自动关闭 MixUp**（避免正则化叠加） |

## 六、模型架构

### 6.1 Teacher: EfficientNet-B4

```
EfficientNet-B4 (timm, pretrained ImageNet)
    ├── backbone (~17M params)
    ├── 中间层 hook → 捕获 stage 特征用于 KD
    └── 分类头: Linear(1792→6)
```

### 6.2 Student: EfficientNet-B0

```
EfficientNet-B0 (timm, pretrained ImageNet)
    ├── backbone (~4M params)
    ├── 中间层 hook → 与 Teacher 对应 stage 对齐
    └── 分类头: Linear(1280→6)
```

### 6.3 知识蒸馏

```
Loss = α × KL(soft_S, soft_T) × T²
     + (1-α) × CE(student, labels)
     + β × MSE(proj(feat_S), feat_T)
```

- T=4, α=0.7, β=0.1
- 特征对齐使用 1×1 卷积投影层

## 七、训练流程

### 7.1 三阶段流水线（70 分钟 GPU 时限适配）

```
Step 1: Train Teacher (15 epochs)                              ~3.5 h
    ├── 损失: FocalLoss (γ=1.0, 动态 α)
    ├── 优化: AdamW (lr=5e-5, wd=1e-4) + SAM (后 5 epoch)
    ├── LR: Linear warmup 2 epoch → CosineAnnealing
    ├── 正则: MixUp α=0.2 (Fast), EMA decay=0.99997
    ├── 策略: DRW (前 9 epoch 标准 → 后 6 epoch cloudy OS 2×)
    ├── 精度: BF16 autocast + clip_grad (无需 GradScaler)
    ├── 备份: results/checkpoints/ 每 epoch 保存, 保留 20 个
    └── 输出: teacher_best.pth (按 EMA F1 择优)

Step 2: Knowledge Distillation (15 epochs)                      ~15 min
    ├── 教师: 加载 teacher_best.pth, 冻结
    ├── 学生: EfficientNet-B0, ImageNet 预训练
    ├── 损失: α×KL + (1-α)×CE + β×FeatureMSE
    ├── 优化器: AdamW (lr=1e-3, wd=1e-4)
    └── 输出: student_distilled_best.pth

Step 3: Structured Pruning + Fine-tune                          ~5 min
    ├── 策略: 渐进式 2 轮 (20% → 40%)
    ├── 方法: L2-norm 结构化通道剪枝 (只剪 1×1 conv)
    ├── 每轮后微调 5 epochs (lr=1e-4)
    └── 输出: student_pruned_final.pth
```

### 7.2 教师策略详解

DRW (Deferred Re-weighting, LDAM NeurIPS 2019):
- 前 60% epoch：标准采样，让模型学好特征表示
- 后 40% epoch：cloudy 过采样 2×，校准决策边界

SAM + MixUp 解耦:
- Fast 阶段（epoch 1-10）：MixUp α=0.2 做输入空间正则化
- SAM 阶段（epoch 11-15）：**自动关闭 MixUp**，仅 OS+扰动
- 原因：SAM+MU 叠加导致过度正则化，实验验证 Val F1 反而下降

### 7.3 已知经验

- **cloudy↔sunny 是最大混淆对**：1510 张 sunny→cloudy，cloudy Precision 仅 0.71
- **thundery/snowy 几乎完美**：F1 > 0.95
- **SAM 仅做收尾**：前 10 epoch Fast 打下基础，SAM 后 5 轮做平坦极小值平滑
- **全量 60k 评估用 num_workers=0**：Windows DataLoader 共享内存有限

## 八、推理与导出

```
PyTorch Model (.pth)
    → torch.onnx.export (opset=13, constant folding)
    → weather_model.onnx (FP32)
    → onnxruntime.quantization.quantize_dynamic (QInt8)
    → weather_model_int8.onnx
    → CPU 推理加速 2-4×，体积 ~50%
```

## 九、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 380 | B4 原生分辨率 |
| `batch_size` | 8 | RTX 5070 8GB 稳妥值 |
| `teacher_lr` | 5e-5 | 保守 LR 保护预训练特征 |
| `focal_gamma` | 1.0 | 降为 1，给 cloudy 梯度 |
| `mixup_alpha` | 0.2 | SAM 阶段自动关闭 |
| `sam_rho` | 0.05 | 后 5 epoch 启用 |
| `ema_decay` | 0.99997 | ~33k steps 平滑窗口 |
| `kd_temperature` | 4.0 | 软标签平滑 |
| `kd_alpha` | 0.7 | 软:硬标签权重比 |
| `label_smoothing` | 0.0 | 关闭，保护 KD 暗知识 |
| `prune_ratio` | 0.4 | 最终剪枝比例 |
| `seed` | 42 | 全局随机种子 |

## 十、预期与实际效果

| 阶段 | 模型 | Macro F1（预期） | Macro F1（实际） | 备注 |
|------|------|----------------:|----------------:|------|
| Teacher | EfficientNet-B4 | ~0.88 | **0.8933** | 全量 60k 评估，超过预期 |
| KD 蒸馏 | B0 distilled | ~0.85 | — | 待训练 |
| KD+剪枝 | B0 pruned 40% | ~0.84 | — | 待训练 |
| ONNX INT8 | ONNX Runtime | ~0.84 | — | 待量化 |
