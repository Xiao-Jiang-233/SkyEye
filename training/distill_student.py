# ============ training/distill_student.py ============
"""
知识蒸馏入口脚本

流程：加载 Teacher → 创建 Student → 初始化 DistillationTrainer → 训练
输出：results/student_distilled_best.pth
"""
import os

import torch

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from models.distill_wrapper import DistillationTrainer
from data.dataset import create_dataloaders


def run_distillation():
    """执行完整知识蒸馏流程"""
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 1) 加载训练好的教师模型
    print(f"Loading teacher: {cfg['teacher_model']} ...")
    teacher = WeatherEfficientNet(
        model_name=cfg["teacher_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,  # 使用自己训练的权重
    ).to(device)
    if not os.path.isfile(cfg["teacher_ckpt"]):
        raise FileNotFoundError(
            f"教师模型不存在: {cfg['teacher_ckpt']}，请先完成教师训练"
        )
    teacher.load_state_dict(torch.load(
        cfg["teacher_ckpt"],
        map_location=device,
        weights_only=True,
    ))
    teacher.eval()
    print("Teacher model loaded and frozen.")

    # 2) 创建学生模型
    print(f"Creating student: {cfg['student_model']} ...")
    student = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=True,  # ImageNet 预训练
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"启用 DataParallel: {torch.cuda.device_count()} 张 GPU")
        student = torch.nn.DataParallel(student)
    print("Student model created.")

    # 3) 数据加载
    train_loader, val_loader, _, class_names = create_dataloaders()

    # 4) 蒸馏训练
    trainer = DistillationTrainer(teacher, student, device, cfg, class_names=class_names)
    distilled_student = trainer.train(train_loader, val_loader)

    print(f"Distillation complete. Model saved to {cfg['distilled_ckpt']}")
    return distilled_student


if __name__ == "__main__":
    run_distillation()
