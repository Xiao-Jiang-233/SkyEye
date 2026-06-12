"""训练与蒸馏共用的 checkpoint 工具。"""

import os
import tempfile

import torch
import torch.nn as nn


def unwrap_model(model):
    """DataParallel 模型返回底层 module，普通模型原样返回。"""
    return model.module if isinstance(model, nn.DataParallel) else model


def atomic_torch_save(payload, path):
    """先写同目录临时文件，再原子替换目标 checkpoint。"""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(fd)

    try:
        torch.save(payload, temp_path)
        if os.path.getsize(temp_path) == 0:
            raise OSError(f"checkpoint 临时文件为空: {temp_path}")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def atomic_save_state_dict(model, path):
    """原子保存解包后的模型 state_dict。"""
    atomic_torch_save(unwrap_model(model).state_dict(), path)
