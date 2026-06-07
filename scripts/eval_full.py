"""在全部 60k 数据集上评估训练好的模型

用法：
    python scripts/eval_full.py                                    # 默认评估 teacher_best.pth (B4)
    python scripts/eval_full.py results/checkpoints/teacher/fast_mu_best.pth
    python scripts/eval_full.py results/student_distilled_best.pth --model efficientnet_b0
    python scripts/eval_full.py results/student_pruned_final.pth   --model efficientnet_b0
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.augmentations import get_val_transforms
from data.dataset import prepare_data


def main():
    parser = argparse.ArgumentParser(description="全量 60k 数据集评估模型")
    parser.add_argument(
        "checkpoint", nargs="?", default=CONFIG["teacher_ckpt"],
        help=f"模型 checkpoint 路径（默认: {CONFIG['teacher_ckpt']}）",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="模型架构（默认: 从路径推断，含 b0→efficientnet_b0，含 b4→efficientnet_b4）",
    )
    args = parser.parse_args()

    # 自动推断模型架构
    if args.model:
        model_name = args.model
    elif "b0" in args.checkpoint.lower() or "student" in args.checkpoint.lower():
        model_name = "efficientnet_b0"
    else:
        model_name = "efficientnet_b4"

    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # 加载模型
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {model_name}")
    model = WeatherEfficientNet(
        model_name=model_name,
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device))
    model.eval()

    # 全量数据集
    data_root = prepare_data()
    full_dataset = ImageFolder(data_root, transform=get_val_transforms(cfg["img_size"]))
    loader = DataLoader(full_dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available())
    class_names = full_dataset.classes
    print(f"Dataset: {len(full_dataset)} images, {len(class_names)} classes: {class_names}")

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 指标
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100

    # 从路径提取模型名用于显示
    display_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"  {model_name} @ {display_name} — 全量 {len(full_dataset)} 张评估")
    print(f"{'='*60}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Accuracy:    {acc:.2f}%")
    print(f"{'='*60}")
    print(f"  Per-Class F1:")
    for name, f1 in zip(class_names, per_class_f1):
        filled = int(f1 * 20)
        bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
        print(f"    {name:<12s}: {f1:.4f}  {bar}")
    print(f"{'='*60}")

    # 分类报告
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names, digits=4)}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (行=真实, 列=预测):")
    header = "        " + "".join(f"{n:>7s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{v:7d}" for v in cm[i])
        print(f"  {name:<6s}{row}")


if __name__ == "__main__":
    main()
