#!/usr/bin/env python
# ============ scripts/local_train.py ============
"""
本地开发脚本 — 分阶段运行训练管线

用法:
    python scripts/local_train.py all         # 运行全部阶段
    python scripts/local_train.py teacher     # 仅训练教师
    python scripts/local_train.py distill     # 仅知识蒸馏
    python scripts/local_train.py prune       # 仅剪枝 + 微调
    python scripts/local_train.py export      # 仅 ONNX 导出 + 量化 + 测速
    python scripts/local_train.py check       # 仅检查环境

对应 Notebook 中的 Cell 2-7.
"""
import sys
import time


def stage_check():
    """Cell 2: 检查环境和配置"""
    print("=" * 50)
    print("Stage 1/6: 环境检查")
    print("=" * 50)

    from config import CONFIG
    import torch

    print(f"Device:       {CONFIG['device']}")
    print(f"CUDA:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:   {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    print(f"Model:        Teacher={CONFIG['teacher_model']}, Student={CONFIG['student_model']}")
    print(f"Classes:      {CONFIG['class_names']}")
    print(f"Image size:   {CONFIG['img_size']}")
    print(f"Batch size:   {CONFIG['batch_size']}")
    print("OK\n")
    return CONFIG


def stage_data():
    """Cell 3: 数据准备"""
    print("=" * 50)
    print("Stage 2/6: 数据准备")
    print("=" * 50)

    from data.dataset import create_dataloaders
    train_loader, val_loader, class_counts = create_dataloaders()

    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"Class distribution: {class_counts.astype(int)}\n")

    return train_loader, val_loader


def stage_teacher():
    """Cell 4: 训练教师模型"""
    print("=" * 50)
    print("Stage 3/6: 训练教师模型 (EfficientNet-B5)")
    print("=" * 50)

    from training.train_teacher import train_teacher
    t_start = time.time()
    teacher = train_teacher()
    elapsed = time.time() - t_start
    print(f"Teacher training done in {elapsed/60:.1f} min\n")
    return teacher


def stage_distill():
    """Cell 5: 知识蒸馏"""
    print("=" * 50)
    print("Stage 4/6: 知识蒸馏 (B5 → B0)")
    print("=" * 50)

    from training.distill_student import run_distillation
    t_start = time.time()
    student = run_distillation()
    elapsed = time.time() - t_start
    print(f"Distillation done in {elapsed/60:.1f} min\n")
    return student


def stage_prune():
    """Cell 6: 结构化剪枝 + 微调"""
    print("=" * 50)
    print("Stage 5/6: 结构化剪枝 + 微调")
    print("=" * 50)

    from training.prune_finetune import prune_and_finetune
    t_start = time.time()
    pruned = prune_and_finetune()
    elapsed = time.time() - t_start
    print(f"Pruning + fine-tune done in {elapsed/60:.1f} min\n")
    return pruned


def stage_export():
    """Cell 7: ONNX 导出 + INT8 量化 + CPU 测速"""
    print("=" * 50)
    print("Stage 6/6: ONNX 导出 + INT8 量化 + CPU 测速")
    print("=" * 50)

    from inference.export_onnx import export_to_onnx, quantize_to_int8, benchmark_cpu
    from config import CONFIG

    onnx_path = export_to_onnx(CONFIG["pruned_ckpt"])
    int8_path = quantize_to_int8(onnx_path)
    results = benchmark_cpu(onnx_path, int8_path)

    print(f"\nFinal results: FP32={results['fp32_ms']:.1f}ms  INT8={results['int8_ms']:.1f}ms  Speedup={results['speedup']:.1f}×")
    return results


STAGES = {
    "check":   stage_check,
    "data":    stage_data,
    "teacher": stage_teacher,
    "distill": stage_distill,
    "prune":   stage_prune,
    "export":  stage_export,
}

ORDER = ["check", "data", "teacher", "distill", "prune", "export"]


def run_from(stage_name: str):
    """从指定 stage 开始运行到结束（如 teacher → distill → prune → export）"""
    start_idx = ORDER.index(stage_name)
    for name in ORDER[start_idx:]:
        fn = STAGES[name]
        fn()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/local_train.py <stage>")
        print(f"Stages: all | {' | '.join(ORDER)}")
        sys.exit(1)

    stage = sys.argv[1]
    t0 = time.time()

    if stage == "all":
        for name in ORDER:
            STAGES[name]()
    elif stage in STAGES:
        run_from(stage)
    else:
        print(f"Unknown stage: {stage}")
        print(f"Available: all | {' | '.join(ORDER)}")
        sys.exit(1)

    total = time.time() - t0
    print(f"\nTotal elapsed: {total/60:.1f} min")


if __name__ == "__main__":
    main()
