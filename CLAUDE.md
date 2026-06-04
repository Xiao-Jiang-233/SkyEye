# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 [Mo 平台](https://momodel.cn) 上的机器学习项目。Mo 平台是一个内嵌 JupyterLab 的在线 IDE，支持 GPU 训练和模型部署。

项目名：**SkyEye**，九类天气图片分类任务（cloudy, dew, foggy, rainy, rime, sandstorm, snowy, sunny, thundery），当前训练启用 6 类（dew/rime/sandstorm 通过 skip_classes 暂缓）。
技术方案：EfficientNet-B5（教师）→ 知识蒸馏 → EfficientNet-B0（学生）→ 结构化剪枝 → ONNX 导出 → INT8 量化。
比赛约束：GPU 训练 → CPU 推理，总时限 70 分钟。
设计文档：`docs/superpowers/specs/2026-06-03-efficientnet-kd-pruning-design.md`

## 开发环境

### Mo 平台（云端 Linux）

- **运行时环境**: Python 3.13.13 | PyTorch 2.8.0+cu128 | CUDA (Mo 平台 GPU)
- 开发方式：纯模块化 `.py` 文件 + `main.ipynb` 作为入口调用，`prepare_datasets.ipynb` 备用
- 模块结构：`config.py`（超参数）→ `data/`（加载+增强）→ `models/`（EfficientNet封装+蒸馏）→ `training/`（教师训练+蒸馏+剪枝微调）→ `inference/`（ONNX导出+INT8量化+CPU推理）→ `utils/`（指标+日志）
- Python 包管理：`!pip install <package>`（在 Notebook cell 中直接运行）
- 运行代码：`Shift + Enter`

### 本地开发（Windows 11）

- **运行时环境**: Python 3.13.x | PyTorch 2.8.0+cu128 | CUDA (RTX 5070)
- **虚拟环境**: `.venv/`（已在 `.gitignore` 中排除），通过 pip 直接安装依赖
- 开发方式：纯模块化 `.py` 文件，通过 `scripts/local_train.py` CLI 运行
- **Windows 特别说明**：
  - `num_workers` 自动设为 `0`（`config.py` 中检测 `sys.platform`），避免 multiprocessing spawn 卡死
  - `pin_memory` 自动检测 CUDA 可用性，CPU 上自动关闭
  - 混合精度（`fp16`）仅在 CUDA 上生效，CPU 上自动跳过
  - 数据集路径 `data_root` 需要根据本地实际情况修改（Mo 平台的 hash 路径在本地不存在）

## 目录结构

| 路径 | 用途 |
|---|---|
| `main.ipynb` | Notebook 入口，分阶段调用各 .py 模块（使用 `!cp -R` / `!7zx` 预处理数据） |
| `prepare_datasets.ipynb` | 数据集准备（备用），日常训练直接用 `main.ipynb` |
| `scripts/local_train.py` | 本地开发 CLI 脚本（分阶段运行训练管线） |
| `scripts/run.sh` | Linux/macOS/Git Bash 快捷启动脚本 |
| `scripts/run.bat` | Windows CMD 快捷启动脚本 |
| `scripts/run.ps1` | Windows PowerShell 快捷启动脚本 |
| `datasets/` | 导入的数据集，**只读**，需复制到其他目录才能修改 |
| `results/` | 训练结果和模型检查点存放处 |
| `results/tb_results/` | TensorBoard 日志存放处 |
| `_OVERVIEW.md` | 项目介绍（功能、环境、结构、流程） |
| `docs/接口文档.md` | 模块 API 接口文档 |

| `app_spec.yml` | 定义模型输入输出，用于部署服务（待创建） |

## 已导入的数据集

### 数据集 1：weather_classification（主数据集）

`datasets/<hash>/weather_classification/` — 6 类 × 各 10,000 张 = 共 **60,000 张**天气图片，按类别分目录存放：

| 目录 | 数量 | 中文 |
| --- | --- | --- |
| `cloudy/` | 10,000 | 多云 |
| `foggy/` | 10,000 | 雾霾 |
| `rainy/` | 10,000 | 雨天 |
| `snowy/` | 10,000 | 雪天 |
| `sunny/` | 10,000 | 晴天 |
| `thundery/` | 10,000 | 雷暴 |

> ⚠️ 类名统一使用形容词形式（cloudy/dew/foggy/rainy/rime/sandstorm/snowy/sunny/thundery），
> 数据集中 `haze`、`snow`、`thunder` 等名词变体通过 `class_aliases` 自动映射。

### 数据集 2：weather-dataset.zip（补充数据集）

`datasets/jehanbhathena/weather-dataset.zip` — 6,862 张，11 个细分类别，通过 `class_aliases` 映射到 9 个目标类（其中 3 类暂缓）：

| 原始类 | 映射到 | 数量 | 状态 |
| --- | --- | --- | --- |
| dew | dew | 698 | ⏭ skip |
| fogsmog | foggy | 851 | ✓ |
| rime | rime | 1,160 | ⏭ skip |
| sandstorm | sandstorm | 692 | ⏭ skip |
| frost, glaze, snow | snowy | 1,735 | ✓ |
| hail, lightning | thundery | 968 | ✓ |
| rain | rainy | 526 | ✓ |
| rainbow | sunny | 232 | ✓ |

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

### Mo 平台（Notebook cell 中执行）

```bash
# 列出已安装的包
!pip list --format=columns

# 检查某个包是否存在
!pip show <package_name>

# 安装包
!pip install <package_name>

# 更新包
!pip install <package_name> --upgrade

# 解压 zip 文件
!7zx file_name.zip

# 复制数据集到可写目录
!cp -R ./datasets/<imported_dataset> ./<your_folder>

# 查看当前目录内容
ls

# TensorBoard 可视化
!tensorboard --logdir results/tb_results/ --bind_all --port 6006

# 本地开发：分阶段运行训练管线
python scripts/local_train.py check     # 检查环境
python scripts/local_train.py teacher   # 训练教师
python scripts/local_train.py distill   # 仅知识蒸馏
python scripts/local_train.py prune     # 仅剪枝 + 微调
python scripts/local_train.py export    # 仅 ONNX 导出 + 量化 + 测速
python scripts/local_train.py all       # 完整管线
```

### 本地开发（Windows）

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
python scripts/local_train.py teacher   # 训练教师（CPU 上会很慢）
python scripts/local_train.py distill   # 仅知识蒸馏
python scripts/local_train.py prune     # 仅剪枝 + 微调
python scripts/local_train.py export    # 仅 ONNX 导出 + 量化 + 测速
python scripts/local_train.py all       # 完整管线

# 单张图片推理
python -m inference.infer <image_path>

# TensorBoard 可视化
tensorboard --logdir results/tb_results/
```

> ⚠️ **Windows 注意**：本地没有 GPU，训练极慢。建议仅在 Windows 上做代码开发和调试，
> 实际训练在 Mo 平台 GPU 环境执行。`data_root` 路径需要根据本地数据集位置修改 `config.py`。
> 数据集自动合并到 `writable_root`（默认 `_data/weather/`），此目录在 `.gitignore` 中已排除。

## 注意事项

- 比赛约束：GPU 训练 + CPU 推理，总时限 70 分钟（epoch 已缩减适配）
- 类名统一使用形容词：数据集目录可能用名词（`haze`, `snow`, `thunder`），`class_aliases` 自动映射到 `foggy`, `snowy`, `thundery`
- `datasets/` 目录是只读的，不可直接修改其中的文件
- **Mo 平台不允许 `.` 开头的文件/目录**，故使用 `_data/` 而非 `.data/`
- 预训练模型下载已配置 HF 镜像：`config.py` 中 `HF_ENDPOINT=https://hf-mirror.com`
- 运行 job 训练时，结果务必指定输出到 `results/` 目录
- 项目部署需创建 `app_spec.yml` 定义输入输出接口
- `.localenv/` 和 `.venv/` 是本地虚拟环境目录（已在 `.gitignore` 中排除）
- **Windows**：`num_workers` 自动设为 0（`config.py` 检测 `sys.platform`），`pin_memory` 自动适配
- **Windows**：`fp16` 混合精度仅在 CUDA 上生效，CPU 训练自动跳过
- **Windows**：训练前需修改 `config.py` 中的 `data_root` 为本地数据集路径

## 核心依赖

基准版本：**torch 2.8.0+cu128 / torchvision 0.23.0+cu128**

```bash
pip install -r requirements.txt
```

Notebook 中直接执行 Cell 1。

| 包 | 版本 |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| timm | 1.0.27 |
| onnx | 1.21.0 |
| onnxruntime | 1.26.0 |
| tqdm / scikit-learn / scipy / tensorboard | latest |

> RTX 5070（Blackwell）驱动 ≥596 支持 CUDA 12.8，与 cu128 索引完全兼容。
