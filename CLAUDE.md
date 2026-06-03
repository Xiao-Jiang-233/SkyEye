# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 [Mo 平台](https://momodel.cn) 上的机器学习项目。Mo 平台是一个内嵌 JupyterLab 的在线 IDE，支持 GPU 训练和模型部署。

项目名：**SkyEye**，目前处于初始模板状态（仅包含平台自动生成的样板文件），推测目标为天气分类相关任务。

## 开发环境

- 主要开发文件：`coding_here.ipynb`（Jupyter Notebook）
- 操作系统：Linux（Mo 平台云端环境）
- Python 包管理：`!pip install <package>`（在 Notebook cell 中直接运行）
- 运行代码：`Shift + Enter`

## 目录结构

| 路径 | 用途 |
|---|---|
| `coding_here.ipynb` | 主开发笔记本，所有代码在此编写 |
| `datasets/` | 导入的数据集，**只读**，需复制到其他目录才能修改 |
| `results/` | 训练结果和模型检查点存放处 |
| `results/tb_results/` | TensorBoard 日志存放处 |
| `_OVERVIEW.md` | 项目介绍（待填写） |
| `app_spec.yml` | 定义模型输入输出，用于部署服务（待创建） |

## 已导入的数据集

`datasets/69f46e75dbb43ba9e05483c1-69e0f1d5638ba61f00d54c83/weather_classification/thunder/` — 包含约 1,728 张雷电天气图片（thunder_00000.jpg 到 thunder_01727.jpg），用于天气分类任务。

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

- `datasets/` 目录是只读的，不可直接修改其中的文件
- 运行 job 训练时，结果务必指定输出到 `results/` 目录
- 项目部署需创建 `app_spec.yml` 定义输入输出接口
- `.localenv/` 是本地虚拟环境目录（已在 `.gitignore` 中排除）
