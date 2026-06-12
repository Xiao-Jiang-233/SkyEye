"""评估训练好的模型

默认仅在验证集（15% holdout）上评估，避免训练样本污染。
使用 --full 可在全部 60k 上评估（含训练样本，仅用于调试）。

用法：
    python scripts/eval_full.py                                    # val set (holdout, 推荐)
    python scripts/eval_full.py --full                             # 全部 60k（含训练样本）
    python scripts/eval_full.py results/student_distilled_best.pth --model efficientnet_b0
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.augmentations import get_val_transforms
from data.dataset import prepare_data


def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument(
        "checkpoint", nargs="?", default=CONFIG["teacher_ckpt"],
        help=f"模型 checkpoint 路径（默认: {CONFIG['teacher_ckpt']}）",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="模型架构（默认: 从路径推断，含 b0→efficientnet_b0，含 b4→efficientnet_b4）",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="在全部 60k 上评估（含训练样本，仅用于调试；默认仅在 15% holdout 上评估）",
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

    # 数据集
    data_root = prepare_data()
    full_dataset = ImageFolder(data_root, transform=get_val_transforms(cfg["img_size"]))
    class_names = full_dataset.classes
    print(f"Dataset: {len(full_dataset)} images, {len(class_names)} classes: {class_names}")

    if args.full:
        eval_dataset = full_dataset
        label = f"全量 {len(full_dataset)} 张"
    else:
        # 分层划分，与训练时一致，仅评估 holdout 验证集
        indices = np.arange(len(full_dataset))
        _, val_idx = train_test_split(
            indices,
            test_size=cfg["val_split"],
            stratify=full_dataset.targets,
            random_state=cfg["seed"],
        )
        eval_dataset = Subset(full_dataset, val_idx)
        label = f"Val holdout {len(val_idx)} 张（{cfg['val_split']:.0%}）"

    # Windows 全量评估需 num_workers=0（共享内存限制）
    nw = 0
    loader = DataLoader(eval_dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=torch.cuda.is_available())
    print(f"Evaluating: {label} (num_workers={nw})")

    # 推理（方案 A：logit bias，与 train_teacher.evaluate 一致）
    all_preds, all_labels = [], []
    bias_cfg = cfg.get("logit_bias", {})
    logit_bias = torch.zeros(cfg["num_classes"], device=device)
    if bias_cfg:
        for cls_name, val in bias_cfg.items():
            if cls_name in class_names:
                logit_bias[class_names.index(cls_name)] = val
        print(f"Logit bias: {bias_cfg}")

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            if bias_cfg:
                logits = logits - logit_bias.unsqueeze(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    # 指标
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100

    # 从路径提取模型名用于显示
    display_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"  {model_name} @ {display_name} — {label}评估")
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
