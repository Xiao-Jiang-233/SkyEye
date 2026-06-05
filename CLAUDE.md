# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

项目名：**SkyEye**，九类天气图片分类任务（cloudy, dew, foggy, rainy, rime, sandstorm, snowy, sunny, thundery），当前训练启用 6 类（dew/rime/sandstorm 通过 skip_classes 暂缓）。
技术方案：EfficientNet-B4（教师）→ 知识蒸馏 → EfficientNet-B0（学生）→ 结构化剪枝 → ONNX 导出 → INT8 量化。
比赛约束：GPU 训练 → CPU 推理，总时限 70 分钟。
设计文档：`docs/superpowers/specs/2026-06-03-efficientnet-kd-pruning-design.md`

## 开发环境

- **操作系统**: Windows 11
- **GPU**: RTX 5070（Blackwell，CUDA 13.0 / cu130）
- **运行时环境**: Python 3.13.x | PyTorch 2.12.0+cu130 | CUDA
- **虚拟环境**: `.venv/`（已在 `.gitignore` 中排除）
- 开发方式：纯模块化 `.py` 文件，`main.ipynb` 作为 Jupyter 入口，`scripts/local_train.py` 作为 CLI 入口
- 模块结构：`config.py`（超参数）→ `data/`（加载+增强）→ `models/`（EfficientNet封装+蒸馏）→ `training/`（教师训练+蒸馏+剪枝微调）→ `inference/`（ONNX导出+INT8量化+CPU推理）→ `utils/`（指标+日志）
- **Windows 特别说明**：
  - `num_workers` 自动设为 2（`config.py` 检测 `sys.platform`），避免 multiprocessing spawn 卡死
  - `pin_memory` 自动检测 CUDA 可用性
  - 混合精度（BF16）仅在 CUDA 上生效，CPU 上自动跳过
  - 数据集路径通过 `config.py` 中的 `data_roots: "auto"` 自动发现

## 目录结构

| 路径                     | 用途                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| `main.ipynb`             | Jupyter Notebook 入口，按阶段调用各 .py 模块                                   |
| `scripts/local_train.py` | CLI 脚本（分阶段运行训练管线）                                                 |
| `scripts/eval_full.py`   | 全量 60k 数据集评估脚本                                                        |
| `scripts/run.sh`         | Linux/macOS/Git Bash 快捷启动脚本                                              |
| `scripts/run.bat`        | Windows CMD 快捷启动脚本                                                       |
| `scripts/run.ps1`        | Windows PowerShell 快捷启动脚本                                                |
| `datasets/`              | 导入的数据集，**只读**，通过 `prepare_data()` 复制到可写目录                   |
| `results/`               | 训练结果和模型检查点存放处                                                     |
| `results/tb_results/`    | TensorBoard 日志存放处                                                         |
| `_OVERVIEW.md`           | 项目介绍，**从 README.md 自动同步**，请勿手动编辑，修改 README.md 后 pull 即可 |
| `docs/接口文档.md`       | 模块 API 接口文档                                                              |
| `app_spec.yml`           | 定义模型输入输出，用于部署服务（待创建）                                       |

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

`datasets/jehanbhathena/weather-dataset.zip` — 6,862 张，11 个细分类别，通过 `class_aliases` 映射到 9 个目标类（其中 3 类暂缓）：

| 原始类             | 映射到    | 数量  | 状态   |
| ------------------ | --------- | ----- | ------ |
| dew                | dew       | 698   | ⏭ skip |
| fogsmog            | foggy     | 851   | ✓      |
| rime               | rime      | 1,160 | ⏭ skip |
| sandstorm          | sandstorm | 692   | ⏭ skip |
| frost, glaze, snow | snowy     | 1,735 | ✓      |
| hail, lightning    | thundery  | 968   | ✓      |
| rain               | rainy     | 526   | ✓      |
| rainbow            | sunny     | 232   | ✓      |

> dew/rime/sandstorm 通过 `skip_classes` 暂缓加载（主数据集无对应类）。移除 `skip_classes` 中条目即可启用。

### 多数据集合并

`config.py` 中 `data_roots` 支持三种模式，`prepare_data()` 自动合并到 `writable_root`。

**① auto 模式（推荐）**：自动扫描 `datasets/` 下所有导入，发现 `weather_classification/` 或 `.zip`，用 `class_aliases` 自动映射类名差异（`haze→foggy`，`snow→snowy`，`thunder→thundery`）：

```python
"data_roots": "auto",
"class_aliases": {
    "haze": "foggy", "fog": "foggy",
    "snow": "snowy", "rain": "rainy",
    "thunder": "thundery", "thunderstorm": "thundery", "lightning": "thundery",
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

# 运行训练管线（三选一）
bash scripts/run.sh check      # Git Bash
scripts\run.bat check          # CMD
.\scripts\run.ps1 check        # PowerShell

# 或直接调用 Python
python scripts/local_train.py check     # 检查环境
python scripts/local_train.py teacher   # 训练教师
python scripts/local_train.py distill   # 仅知识蒸馏
python scripts/local_train.py prune     # 仅剪枝 + 微调
python scripts/local_train.py export    # 仅 ONNX 导出 + 量化 + 测速
python scripts/local_train.py all       # 完整管线

# 单张图片推理
python -m inference.infer <image_path>

# 全量数据集评估
python scripts/eval_full.py

# TensorBoard 可视化
tensorboard --logdir results/tb_results/
```

## 注意事项

- **`_OVERVIEW.md` 是 `README.md` 的镜像文件，修改项目概述时只改 `README.md`，完成后 `cp README.md _OVERVIEW.md` 同步即可**
- 类名统一使用形容词：数据集目录可能用名词（`haze`, `snow`, `thunder`），`class_aliases` 自动映射到 `foggy`, `snowy`, `thundery`
- `datasets/` 目录是只读的，不可直接修改其中的文件
- 预训练模型下载已配置 HF 镜像：`config.py` 中 `HF_ENDPOINT=https://hf-mirror.com`
- 训练结果务必指定输出到 `results/` 目录
- `.venv/` 是本地虚拟环境目录（已在 `.gitignore` 中排除）
- **Windows**：`num_workers` 自动适配（`config.py` 检测 `sys.platform`），`pin_memory` 自动适配
- **Windows**：`fp16`/BF16 混合精度仅在 CUDA 上生效，CPU 训练自动跳过
- BF16 autocast 用于训练（RTX 5070 原生支持，无需 GradScaler）

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
