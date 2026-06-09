# EfficientNet + 知识蒸馏 + 结构化剪枝 — 设计文档（修订版）

> 日期：2026-06-08（修订） | 项目：SkyEye | 赛题：智海算法调优 | 状态：教师训练已完成

## 一、比赛规则（来源：官方规则文件 + FAQ）

| 项目 | 内容 | 来源 |
|------|------|------|
| 任务 | 天气图片分类 | 规则文件 §六 |
| 类别 | FAQ 说 4 类，但 MO 平台测试集实际 6 类 → **先保持 6 类** | Q28/Q35/Q44 |
| 评分 | **Macro F1 × 100**，同分按推理速度排名 | 规则文件 §七.1, Q40 |
| 推理 | **CPU**，测试集几千张，总时限 **≤ 70 分钟**（整体，非单张） | Q30, Q37, Q41, Q60 |
| 训练 | **本地不限时**，提交推理代码 + 权重即可 | Q46, Q52 |
| 提交 | 仅一次提交机会，无实时排行榜 | Q27, Q48 |
| 框架 | 推荐 PyTorch，可用预训练模型，可安装第三方库 | Q36, Q53, Q59 |
| 违规 | 禁止外接大模型 API，可用大模型辅助编程 | Q54 |

### 评分与排名规则（规则文件 §七.4）

```
1. 最终得分 = Macro F1 × 100（省赛系统自动评分）
2. 同分 → 模型推理时间由短到长排序
3. 仍相同 → 代码执行效率（内存占用、CPU/GPU 利用率）
4. 仍相同 → 代码规范性（注释完整性、结构清晰度）
```

### 关键时间线

| 节点 | 时间 |
|------|------|
| 报名截止 | 2026-06-15 |
| 预选赛（省赛）| 2026-06-25 ~ 06-27（3 天，含训练） |
| 数据集发布 | 开赛提供 |
| 初赛成绩 | 7 月初 |

## 二、与上一版设计的核心变化

| 变化项 | 旧假设 | 新认识 | 影响 |
|--------|-------|--------|------|
| 参数限制 | ≤ 3M（自定）| **无硬性限制** | 剪枝从刚需降级为速度优化项 |
| 70 分钟 | GPU 训练时限（误读）| **CPU 推理总时限** | 训练不限时，推理按几千张整体计时 |
| 训练 | 必须在时限内完成 | 本地 RTX 5070 **不限时** | 可放心增加 epoch、尝试更强策略 |
| 评分 | 仅看 Macro F1 | Macro F1 × 100 + **速度 tiebreaker** | 推理速度从无关紧要变成第二排序指标 |

## 三、类别架构（保持不变）

`class_names` 共 7 类（6 核心 + `other` 兜底），当前训练 6 类：

| 类 | 训练 | 数据量 | 说明 |
|---|---|---|---|
| cloudy | ✓ | 10,000 | 多云 — 最大混淆源 |
| foggy | ✓ | 10,000 | 雾霾 |
| rainy | ✓ | 10,000 | 雨天 |
| snowy | ✓ | 10,000 | 雪天 — F1 > 0.95 |
| sunny | ✓ | 10,000 | 晴天 — 1478 张误判为 cloudy（已从 2004 改善） |
| thundery | ✓ | 10,000 | 雷暴 — F1 > 0.97 |
| other | ⏭ skip | ~2,550 | dew/rime/sandstorm → 归入 other |

> Q28/Q35 说 4 类（多云/雨天/雪天/晴天），但 Q44 指出 MO 平台实际有 6 类测试集。为安全起见保持 6 类，确认后再调整。

## 四、技术方案

### 4.1 整体流程

```
EfficientNet-B4 (教师, 380px)
    │ FocalLoss + MixUp + EMA + DRW + ConfusionPenalty + LogitAdj
    ▼
teacher_best.pth (当前: Macro F1 0.8941)
    │ 知识蒸馏 (KL + Feature MSE, T=4, α=0.7)
    ▼
EfficientNet-B0 (学生, 蒸馏后)
    │ ONNX 导出 (opset=13) → INT8 动态量化
    ▼
weather_model_int8.onnx → CPU 推理
```

**剪枝变为可选**：如果 INT8 B0 推理速度已满足要求，不剪枝（省精度）。仅在需要争速度排名时做轻量剪枝。

### 4.2 剪枝策略调整

| 项 | 旧 | 新 |
|----|----|----|
| 是否必要 | 是（为 ≤3M）| **否**（无参数限制）|
| 触发条件 | 默认执行 | **仅当 INT8 B0 推理速度不够时** |
| 剪枝率 | 40% | **20-25%**（轻量，保精度） |
| 方式 | 渐进式 2 轮 | **一次性**（中等剪枝率下一次性和渐进等价）|

原因：
1. F1 是主排序指标，剪枝丢的每一分都可能直接拉低排名
2. INT8 量化已大幅加速 CPU 推理，B0 ~4M 原样部署没有问题
3. 速度 tiebreaker 只在 F1 相同时触发，优先级低

### 4.3 推理速度优化（tiebreaker 相关）

即使不剪枝，以下手段也能有效加速 CPU 推理：

- **INT8 动态量化**：2-4× 加速，精度损失极小
- **ONNX Runtime 线程数调优**：匹配评测 CPU 核数
- **批处理推理**：一次喂多张图，减少 Python 循环和 ONNX session 调用开销
- **图像预处理管线化**：numpy/opencv 批量 Resize→CenterCrop→Normalize
- 如果还不够：**轻量剪枝 20-25%**（一次性 L2 结构化剪枝 + 10 epoch 微调）

## 五、训练策略（保持不变）

### 5.1 教师训练（EfficientNet-B4, 380px）

训练不限时，当前 15 epoch 方案（P1: 9 + P2: 6）。SAM 已移除（实验证明导致 F1 从 0.8931 跌至 0.8744）。

| 阶段 | Epochs | 策略 |
|------|--------|------|
| Phase 1 | 1-9 | 标准采样 + MixUp α=0.2 + Per-Class Label Smoothing |
| Phase 2 | 10-15 | DRW cloudy 2× + sunny 2× + ConfusionPenaltyLoss + Per-Class Smoothing |

**sunny→cloudy 混淆四方案**：

| 方案 | 实现 | 生效位置 |
|------|------|----------|
| A: Logit Adjustment | `logit_bias: {sunny: -0.5, cloudy: 0.3}` | evaluate() + eval_full |
| B: Cost-Sensitive Loss | ConfusionPenaltyLoss 包装 FocalLoss，惩罚 sunny→cloudy | Phase 2 |
| C: Sunny Oversampling | Phase 2 对 sunny 也做 2× 过采样 | Phase 2 数据加载 |
| D: Per-Class Smoothing | sunny/cloudy 用 ε=0.1，其他类为 0 | Phase 1+2 FocalLoss |

效果：sunny→cloudy 混淆从 2004 降至 1478（-26%），sunny recall 0.78→0.84，Macro F1 0.8931→0.8941。

### 5.2 知识蒸馏（B4 → B0）

```
Loss = α × KL(soft_S, soft_T) × T²
     + (1-α) × CE(student, labels)
     + β × MSE(proj(feat_S), feat_T)
```

参数：T=4, α=0.7, β=0.1, epochs=15, lr=1e-3

### 5.3 可选：轻量剪枝（仅当需要时）

```
加载蒸馏 B0 → 一次性 L2 结构化剪枝 25% → 微调 10 epoch → 固化 → ONNX INT8
```

## 六、推理与导出

```
PyTorch B0 (.pth)
  → torch.onnx.export (opset=13, dynamic_batch=True)
  → weather_model.onnx (FP32)
  → onnxruntime.quantization.quantize_dynamic (QInt8)
  → weather_model_int8.onnx
```

## 七、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 380 | B4 原生分辨率 |
| `batch_size` | 8 | RTX 5070 8GB |
| `num_classes` | 6 | cloudy/foggy/rainy/snowy/sunny/thundery |
| `teacher_lr` | 5e-5 | 保守 LR |
| `focal_gamma` | 1.0 | cloudy 梯度 |
| `mixup_alpha` | 0.2 | Phase 1+2 全程开启 |
| `per_class_label_smoothing` | sunny/cloudy=0.1 | 方案 D：易混淆类更高平滑 |
| `logit_bias` | sunny=-0.5, cloudy=+0.3 | 方案 A：推理时调整判定门槛 |
| `confusion_penalty_weight` | 0.3 | 方案 B：sunny→cloudy 额外惩罚 |
| `sam_rho` | 0.05 | 后 5 epoch |
| `ema_decay` | 0.99997 | ~33k steps |
| `kd_temperature` | 4.0 | |
| `kd_alpha` | 0.7 | |
| `label_smoothing` | 0.0 | |
| `prune_ratio` | 0.25（可选）| 仅速度需要时启用 |
| `prune_iterations` | 1（可选）| 一次性剪枝 |

## 八、预期与实际效果

| 阶段 | 模型 | Macro F1（预期） | Macro F1（实际） | 备注 |
|------|------|:---:|:---:|------|
| Teacher | EfficientNet-B4 | ~0.88 | **0.8941** | 已完成 |
| KD 蒸馏 | B0 distilled | ~0.85 | — | 待训练 |
| +INT8 量化 | ONNX INT8 | ~0.85 | — | 精度基本无损 |
| +轻量剪枝(可选) | B0 pruned 25% | ~0.84 | — | 仅速度需要时启用 |

## 九、目录结构（不变）

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
│   ├── train_teacher.py            # 教师训练（FocalLoss+MixUp+EMA+DRW+ConfusionPenalty+LogitAdj）
│   ├── distill_student.py          # 知识蒸馏
│   └── prune_finetune.py           # 剪枝 + 微调（可选）
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
