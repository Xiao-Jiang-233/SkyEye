# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

项目名：**SkyEye**，天气图片分类任务。`class_names` 共 7 类（6 核心天气类 + `other` 兜底类），当前训练 6 类（`skip_classes: ["other"]`）。
dew/rime/sandstorm 通过 `class_aliases` 映射到 `other`，暂不参与训练（补充数据集仅 ~700~1200 张，样本量不足）。
技术方案：EfficientNet-B4（教师）→ 知识蒸馏 → EfficientNet-B0（学生）→ ONNX 导出 → INT8 量化。
比赛约束：GPU 训练 → CPU 推理，推理总时限 70 分钟（训练本地不限时）。评分 Macro F1 × 100，同分按推理速度排名。规则详见 [docs/competition-rules.md](docs/competition-rules.md) 和 [docs/competition-faq.md](docs/competition-faq.md)。
当前教师最优：**Macro F1 0.9015 / Acc 89.83%**（全量 60k 评估），瓶颈在 cloudy↔sunny 混淆。
设计文档：[docs/design-efficientnet-kd-pruning.md](docs/design-efficientnet-kd-pruning.md)

## 开发环境

- **操作系统**: Windows 11
- **CPU**: AMD Ryzen 9 9955HX（16 核 32 线程）
- **GPU**: RTX 5070（Blackwell，CUDA 13.0 / cu130）
- **运行时环境**: Python 3.13.x | PyTorch 2.12.0+cu130 | CUDA
- **虚拟环境**: `.venv/`（已在 `.gitignore` 中排除）
- 开发方式：纯模块化 `.py` 文件，`main.ipynb` 作为 Jupyter 入口
- 模块结构：`config.py`（超参数）→ `data/`（加载+增强）→ `models/`（EfficientNet封装+蒸馏）→ `training/`（教师训练+蒸馏）→ `inference/`（ONNX导出+INT8量化+CPU推理）→ `utils/`（指标+日志）
- **Windows 特别说明**：
  - `num_workers` 自动设为 2（`config.py` 检测 `sys.platform`），避免 multiprocessing spawn 卡死
  - `pin_memory` 自动检测 CUDA 可用性
  - 混合精度（BF16）仅在 CUDA 上生效，CPU 上自动跳过
  - 数据集路径通过 `config.py` 中的 `data_roots: "auto"` 自动发现
  - OpenMP/MKL 线程数按 CPU 核数自适应（32 核 → 8 线程/worker）

## 目录结构

| 路径                     | 用途                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| `main.ipynb`             | Jupyter Notebook 入口，按阶段调用各 .py 模块                                   |
| `scripts/eval_full.py`   | 模型评估脚本（默认 holdout val，--full 切换全量）                              |
| `scripts/search_bias.py` | Logit bias 网格搜索脚本（预计算 logits，秒级完成）                             |
| `datasets/`              | 导入的数据集，**只读**，通过 `prepare_data()` 复制到可写目录                   |
| `results/`               | 训练结果和模型检查点存放处                                                     |
| `results/checkpoints/`   | 每 epoch 周期备份（保留最近 20 个，自动滚动清理）                               |
| `results/tb_results/`    | TensorBoard 日志存放处                                                         |
| `_OVERVIEW.md`           | 项目介绍，**从 README.md 自动同步**，请勿手动编辑，修改 README.md 后 pull 即可 |
| `docs/接口文档.md`                        | 模块 API 接口文档                                                              |
| `docs/design-efficientnet-kd-pruning.md`  | 技术方案设计文档                                                               |
| `docs/plan-efficientnet-kd-pruning.md`    | 实施计划                                                                       |
| `docs/competition-rules.md`               | 比赛规则文件                                                                   |
| `docs/competition-faq.md`                 | 比赛 FAQ                                                                       |
| `app_spec.yml`                            | 定义模型输入输出，用于部署服务（待创建）                                       |

## 已导入的数据集

### 数据集 1：weather_classification（主数据集）

`datasets/<hash>/weather_classification/` — 6 类 × 各 10,000 张 = 共 **60,000 张**天气图片，按类别分目录存放：

| 目录        | 数量   | 中文 |
| ----------- | ------ | ---- |
| `cloudy/`   | 10,000 | 多云 |
| `foggy/`    | 10,000 | 雾霾 |
| `rainy/`    | 10,000 | 雨天 |
| `snowy/`    | 10,000 | 雪天 |
| `sunny/`    | 10,000 | 晴天 |
| `thundery/` | 10,000 | 雷暴 |

> ⚠️ 类名统一使用形容词形式（cloudy/dew/foggy/rainy/rime/sandstorm/snowy/sunny/thundery），
> 数据集中 `haze`、`snow`、`thunder` 等名词变体通过 `class_aliases` 自动映射。

### 数据集 2：weather-dataset.zip（补充数据集）

`datasets/jehanbhathena/weather-dataset.zip` — 6,862 张，11 个细分类别，通过 `class_aliases` 映射到 7 个目标类：

| 原始类             | 映射到    | 数量  |
| ------------------ | --------- | ----- |
| dew                | other     | 698   |
| fogsmog            | foggy     | 851   |
| rime               | other     | 1,160 |
| sandstorm          | other     | 692   |
| frost, glaze, snow | snowy     | 1,735 |
| hail, lightning    | thundery  | 968   |
| rain               | rainy     | 526   |
| rainbow            | sunny     | 232   |

> dew/rime/sandstorm 通过 `class_aliases` 映射到 `other`，该类别在 `skip_classes` 中暂不训练。
> 如需启用，将 `"other"` 从 `skip_classes` 移除，`num_classes` 会自动更新。

### 多数据集合并

`config.py` 中 `data_roots` 支持三种模式，`prepare_data()` 自动合并到 `writable_root`。

**① auto 模式（推荐）**：自动扫描 `datasets/` 下所有导入，发现 `weather_classification/` 或 `.zip`，用 `class_aliases` 自动映射类名差异（`haze→foggy`，`snow→snowy`，`thunder→thundery`）：

```python
"data_roots": "auto",
"class_aliases": {
    "haze": "foggy", "fog": "foggy",
    "snow": "snowy", "rain": "rainy",
    "thunder": "thundery", "thunderstorm": "thundery", "lightning": "thundery",
    "dew": "other", "rime": "other", "sandstorm": "other",
},
```

**② 列表模式（手动）**：精确控制每个数据源：

```python
"data_roots": [
    "datasets/<hash1>/weather_classification",
    {"path": "datasets/<hash2>/data_split.zip", "class_map": {"haze": "foggy", "snow": "snowy"}},
],
```

**③ 旧版兼容**：仍支持 `"data_root": "..."` 单路径。

合并特性：
- 逐类逐文件复制，同名跳过（保留先导入的），自动处理 `datasets/` 只读限制
- `.zip` 自动解压，识别单层壳目录；类名自动匹配（精确 → 别名 → 模糊）
- 无法匹配的类别自动跳过并警告；缺失数据源自动跳过

## 常用命令

```bash
# 创建虚拟环境（首次）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate        # CMD
.venv\Scripts\Activate.ps1    # PowerShell
source .venv/Scripts/activate # Git Bash

# 安装依赖
pip install -r requirements.txt

# 单张图片推理
python -m inference.infer <image_path>

# 模型评估（默认 holdout val 15%）
python scripts/eval_full.py

# 全量 60k 评估（含训练样本，仅调试）
python scripts/eval_full.py --full

# Logit bias 网格搜索（无需重新训练）
python scripts/search_bias.py

# TensorBoard 可视化
tensorboard --logdir results/tb_results/
```

## 训练策略

### 教师模型优化组合

训练分 2 阶段，共 15 epoch（P1: 9 + P2: 6），无 SAM。

| 策略 | 参数 | 说明 |
|---|---|---|
| FocalLoss | γ=1.0 | 处理类别不平衡，让困难样本拿到梯度 |
| MixUp | α=0.2 | Zhang et al. (ICLR 2018)，输入空间正则化 |
| EMA | decay=0.99997 | 权重指数滑动平均，平滑窗口 ~33k steps |
| BF16 AMP | autocast + clip_grad | RTX 5070 原生支持，无需 GradScaler |
| DRW 延迟过采样 | P2 阶段 sunny 2× | LDAM (NeurIPS 2019)，先学特征再校准边界；cloudy 过采样已关闭（精度低） |
| Per-Class Label Smoothing | sunny/cloudy ε=0.1，其他 0 | 对易混淆类用更高平滑值（方案 D） |
| ConfusionPenaltyLoss | sunny/rainy/foggy→cloudy λ=0.3 | 对 cloudy 收容方向施加额外惩罚（方案 B） |
| Logit Adjustment | cloudy=+0.55, foggy=-0.25 | 推理时调整判定门槛，bias 网格搜索最优（方案 A）；公式 logits - bias |

### 已知经验

- **SAM 对该任务无效**：SAM rho=0.05 导致 Macro F1 从 0.8931 跌到 0.8744，sunny recall 和 cloudy precision 双降，训练时间翻倍。已移除
- **cloudy↔sunny 是最大混淆对**：sunny→cloudy 1350（13.5%），rainy→cloudy 1248（12.5%），foggy→cloudy 1231（12.3%）。cloudy 精度仅 0.6877，是核心瓶颈
- **thundery/snowy 几乎完美**：F1 > 0.96，特征鲜明
- **MixUp 阶段 Train F1 < Val F1 是正常现象**：因为 Train F1 在混合样本上计算但只与 labels_a 比较，不是 bug

## 注意事项

- **`_OVERVIEW.md` 是 `README.md` 的镜像文件，修改项目概述时只改 `README.md`，完成后 `cp README.md _OVERVIEW.md` 同步即可**
- 类名统一使用形容词：数据集目录可能用名词（`haze`, `snow`, `thunder`），`class_aliases` 自动映射到 `foggy`, `snowy`, `thundery`
- `datasets/` 目录是只读的，不可直接修改其中的文件
- 预训练模型下载已配置 HF 镜像：`config.py` 中 `HF_ENDPOINT=https://hf-mirror.com`
- 训练结果务必指定输出到 `results/` 目录
- `.venv/` 是本地虚拟环境目录（已在 `.gitignore` 中排除）
- `_data/` 是数据集合并可写目录（已在 `.gitignore` 中排除）
- **Windows**：`num_workers` 自动适配（`config.py` 检测 `sys.platform`），`pin_memory` 自动适配
- **Windows**：`fp16`/BF16 混合精度仅在 CUDA 上生效，CPU 训练自动跳过
- **Windows**：全量评估时设置 `num_workers=0`，避免 60k 图 DataLoader 共享内存耗尽
- BF16 autocast 用于训练（RTX 5070 原生支持，无需 GradScaler）
- TensorBoard 仅使用 SCALARS 标签页（loss/F1/Acc/per-class F1），无 GRAPHS/PROFILE/HISTOGRAMS
- 四方案配置均在 `config.py` 中：`logit_bias`（推理门槛，最优 `cloudy=0.55, foggy=-0.25`）、`confusion_penalty_weight`（混淆惩罚，三方向 sunny/rainy/foggy→cloudy）、`per_class_label_smoothing`（按类平滑）
- `active_class_names` = `class_names` - `skip_classes`，推理/ONNX/bias 构建必须用此而非 `class_names`（否则维度错位）
- 教师模型最终保存到 `results/teacher_best.pth`（Phase 依赖文件在 `results/checkpoints/teacher/`）
- **Windows**：PowerShell 命令（`Test-Path`、`Remove-Item` 等）需用 `PowerShell` 工具，不要通过 `Bash` 执行
- **NotebookEdit**：`git checkout`/`git stash` 后需先重新 `Read` notebook，否则会报 "modified since read" 错误
- **nn.Module 损失函数**：带 buffer 的损失（如 ConfusionPenaltyLoss）需 `.to(device)`，`register_buffer` 确保 buffer 随 module 移动
- **分批 commit**：同一文件跨批次改动交织时，`git add -p` 处理不了。正确方式：备份最终状态 → reset → 逐批手动编辑 → 逐批 commit

## 核心依赖

基准版本：**torch 2.12.0 / torchvision 0.27.0**

```bash
pip install -r requirements.txt
```

| 包                                        | 版本   |
| ----------------------------------------- | ------ |
| torch                                     | 2.12.0 |
| torchvision                               | 0.27.0 |
| timm                                      | 1.0.27 |
| onnx                                      | 1.21.0 |
| onnxruntime                               | 1.26.0 |
| tqdm / scikit-learn / scipy / tensorboard | latest |

> RTX 5070（Blackwell）驱动 ≥596 支持 CUDA 13.0，与 cu130 索引完全兼容。
