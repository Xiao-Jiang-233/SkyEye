# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 [Mo 平台](https://momodel.cn) 上的机器学习项目。Mo 平台是一个内嵌 JupyterLab 的在线 IDE，支持 GPU 训练和模型部署。

项目名：**SkyEye**，六类天气图片分类任务（cloudy, haze, rainy, snow, sunny, thunder）。
技术方案：EfficientNet-B5（教师）→ 知识蒸馏 → EfficientNet-B0（学生）→ 结构化剪枝 → ONNX 导出。
设计文档：`docs/superpowers/specs/2026-06-03-efficientnet-kd-pruning-design.md`

## 开发环境

- **运行时环境**: Python 3.9.5 | PyTorch 2.3.1 | CUDA (Mo 平台 GPU)
- 开发方式：纯模块化 `.py` 文件 + `coding_here.ipynb` 作为入口调用
- 模块结构：`config.py`（超参数）→ `data/`（加载+增强）→ `models/`（EfficientNet封装+蒸馏）→ `training/`（教师训练+蒸馏+剪枝微调）→ `inference/`（ONNX导出+推理）→ `utils/`（指标+日志）
- 操作系统：Linux（Mo 平台云端环境）
- Python 包管理：`!pip install <package>`（在 Notebook cell 中直接运行）
- 运行代码：`Shift + Enter`

## 目录结构

| 路径 | 用途 |
|---|---|
| `coding_here.ipynb` | Notebook 入口，分阶段调用各 .py 模块 |
| `datasets/` | 导入的数据集，**只读**，需复制到其他目录才能修改 |
| `results/` | 训练结果和模型检查点存放处 |
| `results/tb_results/` | TensorBoard 日志存放处 |
| `_OVERVIEW.md` | 项目介绍（待填写） |
| `app_spec.yml` | 定义模型输入输出，用于部署服务（待创建） |

## 已导入的数据集

`datasets/<hash>/weather_classification/` — 6 类 × 各 10,000 张 = 共 **60,000 张**天气图片，按类别分目录存放：

| 目录 | 数量 | 中文 |
| --- | --- | --- |
| `cloudy/` | 10,000 | 多云 |
| `haze/` | 10,000 | 雾霾 |
| `rainy/` | 10,000 | 雨天 |
| `snow/` | 10,000 | 雪天 |
| `sunny/` | 10,000 | 晴天 |
| `thunder/` | 10,000 | 雷暴 |

> ⚠️ 数据集中类别目录名为 `snow`（非 `snowy`），代码中统一使用 `snow`。

## 常用命令

在 Notebook cell 中执行：

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
```

## 注意事项

- `snow` 类别：数据集中目录名为 `snow`，代码中统一使用 `snow`（非 `snowy`）
- `datasets/` 目录是只读的，不可直接修改其中的文件
- 运行 job 训练时，结果务必指定输出到 `results/` 目录
- 项目部署需创建 `app_spec.yml` 定义输入输出接口
- `.localenv/` 是本地虚拟环境目录（已在 `.gitignore` 中排除）

## 核心依赖

目标环境：**Python 3.9.5 | PyTorch 2.3.1**

在 Notebook cell 中执行（或直接用 `requirements.txt`）：

```bash
!pip install -r requirements.txt
```

版本清单：

```bash
torch==2.3.1  torchvision==0.18.1  timm==1.0.8  onnx==1.16.1  onnxruntime-gpu==1.18.1  tqdm  scikit-learn
