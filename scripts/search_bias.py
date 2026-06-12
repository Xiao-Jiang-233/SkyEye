"""Logit Bias 网格搜索 — 在 9k holdout 验证集上搜索最优 bias，无需重新训练。

先预计算全部 logits（一次前向），然后纯 tensor 操作搜索 → 几千个组合秒级完成。

两阶段搜索：
  Stage 1: cloudy + sunny 粗搜索（rainy/foggy=0）
  Stage 2: 取 Stage 1 最优，精调 rainy/foggy

用法：
    python scripts/search_bias.py                           # 默认教师模型
    python scripts/search_bias.py --ckpt results/teacher_best.pth --model efficientnet_b4
"""
import argparse
import sys
import json
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet
from data.augmentations import get_val_transforms
from data.dataset import prepare_data


def compute_macro_f1(logits, labels, bias_tensor):
    """纯 tensor 操作：logits - bias → argmax → macro F1。返回 (macro_f1, per_class_f1)。"""
    adjusted = logits - bias_tensor.unsqueeze(0)
    preds = adjusted.argmax(dim=1).cpu().numpy()
    macro_f1 = f1_score(labels, preds, average='macro')
    per_class = f1_score(labels, preds, average=None)
    return macro_f1, per_class


def main():
    parser = argparse.ArgumentParser(description="Logit Bias 网格搜索")
    parser.add_argument("--ckpt", default=CONFIG["teacher_ckpt"],
                        help=f"模型 checkpoint（默认: {CONFIG['teacher_ckpt']}）")
    parser.add_argument("--model", "-m", default="efficientnet_b4",
                        help="模型架构（默认: efficientnet_b4）")
    parser.add_argument("--device", default=CONFIG["device"],
                        help=f"设备（默认: {CONFIG['device']}）")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Architecture: {args.model}")
    print(f"Formula: logits - bias  (bias>0 → harder, bias<0 → easier)")

    # ---- 加载模型 ----
    model = WeatherEfficientNet(
        model_name=args.model,
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location=device))
    model.eval()

    # ---- 加载 holdout 验证集 ----
    data_root = prepare_data()
    full_dataset = ImageFolder(data_root, transform=get_val_transforms(cfg["img_size"]))
    class_names = full_dataset.classes
    print(f"Data: {len(full_dataset)} images, {len(class_names)} classes: {class_names}")

    indices = np.arange(len(full_dataset))
    _, val_idx = train_test_split(
        indices, test_size=cfg["val_split"],
        stratify=full_dataset.targets, random_state=cfg["seed"],
    )
    val_ds = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    print(f"Holdout val: {len(val_ds)} images ({cfg['val_split']:.0%})")

    # ---- 预计算全部 logits + labels（仅一次模型前向） ----
    print("Pre-computing all logits ...")
    all_logits, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Forward"):
            logits = model(images.to(device)).cpu()
            all_logits.append(logits)
            all_labels_list.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    logits_cache = torch.cat(all_logits, dim=0)  # (N, C)
    labels_cache = torch.cat(all_labels_list, dim=0).cpu().numpy()  # (N,)
    print(f"Cached: logits {list(logits_cache.shape)}, labels {labels_cache.shape}")

    # 构建 class_name → index 映射
    cls_idx = {name: i for i, name in enumerate(class_names)}

    # ---- baseline (bias=0) ----
    zero_bias = torch.zeros(len(class_names))
    base_f1, base_per = compute_macro_f1(logits_cache, labels_cache, zero_bias)
    base_per_dict = dict(zip(class_names, base_per))
    print(f"Baseline (bias=0): Macro F1={base_f1:.4f}, cloudy F1={base_per_dict.get('cloudy',0):.4f}")

    # ---- Stage 1: cloudy + sunny 密集搜索 ----
    cloudy_values = np.arange(0.0, 1.05, 0.05)      # 0.0, 0.05, ..., 1.0
    sunny_values = np.arange(-1.0, 0.05, 0.05)       # -1.0, -0.95, ..., 0.0

    print(f"\n{'='*60}")
    print(f"  Stage 1: cloudy × sunny 密集网格 ({len(cloudy_values)}×{len(sunny_values)}={len(cloudy_values)*len(sunny_values)} combos)")
    print(f"  cloudy: [{cloudy_values[0]:.2f}, {cloudy_values[-1]:.2f}]  sunny: [{sunny_values[0]:.2f}, {sunny_values[-1]:.2f}]")
    print(f"{'='*60}")

    stage1 = []
    best_s1_f1, best_s1_bias = 0.0, {}
    total = len(cloudy_values) * len(sunny_values)
    pbar = tqdm(itertools.product(cloudy_values, sunny_values), total=total, desc="Stage 1")

    for cv, sv in pbar:
        bias = torch.zeros(len(class_names))
        bias[cls_idx["cloudy"]] = cv
        bias[cls_idx["sunny"]] = sv
        mf1, per = compute_macro_f1(logits_cache, labels_cache, bias)
        cf1 = per[cls_idx["cloudy"]]
        stage1.append({"cv": round(cv, 2), "sv": round(sv, 2), "mf1": round(mf1, 6), "cf1": round(cf1, 6)})
        if mf1 > best_s1_f1:
            best_s1_f1 = mf1
            best_s1_bias = {"cloudy": round(cv, 2), "sunny": round(sv, 2)}
            best_s1_per = dict(zip(class_names, per))
            pbar.set_postfix({"best": f"{mf1:.4f}", "cF1": f"{cf1:.4f}", "c": cv, "s": sv})

    stage1.sort(key=lambda x: x["mf1"], reverse=True)
    print(f"\nStage 1 Top 10:")
    print(f"{'Rank':<5} {'cloudy':>8} {'sunny':>8} {'Macro F1':>10} {'cloudy F1':>10}")
    for i, r in enumerate(stage1[:10]):
        print(f"{i+1:<5} {r['cv']:>8.2f} {r['sv']:>8.2f} {r['mf1']:>10.4f} {r['cf1']:>10.4f}")

    # ---- Stage 2: rainy + foggy 精调 ----
    rainy_values = np.arange(-0.3, 0.35, 0.05)        # -0.3, -0.25, ..., 0.3
    foggy_values = np.arange(-0.3, 0.35, 0.05)

    best_c = best_s1_bias["cloudy"]
    best_s = best_s1_bias["sunny"]

    print(f"\n{'='*60}")
    print(f"  Stage 2: rainy × foggy 精调 @ cloudy={best_c}, sunny={best_s}")
    print(f"  ({len(rainy_values)}×{len(foggy_values)}={len(rainy_values)*len(foggy_values)} combos)")
    print(f"{'='*60}")

    stage2 = []
    best_s2_f1, best_s2_bias = best_s1_f1, best_s1_bias
    best_s2_per = best_s1_per
    total2 = len(rainy_values) * len(foggy_values)
    pbar2 = tqdm(itertools.product(rainy_values, foggy_values), total=total2, desc="Stage 2")

    for rv, fv in pbar2:
        bias = torch.zeros(len(class_names))
        bias[cls_idx["cloudy"]] = best_c
        bias[cls_idx["sunny"]] = best_s
        bias[cls_idx["rainy"]] = rv
        bias[cls_idx["foggy"]] = fv
        mf1, per = compute_macro_f1(logits_cache, labels_cache, bias)
        cf1 = per[cls_idx["cloudy"]]
        stage2.append({"cv": best_c, "sv": best_s, "rv": round(rv, 2), "fv": round(fv, 2),
                        "mf1": round(mf1, 6), "cf1": round(cf1, 6)})
        if mf1 > best_s2_f1:
            best_s2_f1 = mf1
            best_s2_bias = {"cloudy": best_c, "sunny": best_s, "rainy": round(rv, 2), "foggy": round(fv, 2)}
            best_s2_per = dict(zip(class_names, per))
            pbar2.set_postfix({"best": f"{mf1:.4f}", "cF1": f"{cf1:.4f}", "r": rv, "f": fv})

    stage2.sort(key=lambda x: x["mf1"], reverse=True)

    # 过滤零值 bias 的打印（rainy=0, foggy=0 等同于 stage 1 结果）
    print(f"\nStage 2 Top 10 (非零 rainy/foggy):")
    print(f"{'Rank':<5} {'cloudy':>8} {'sunny':>8} {'rainy':>8} {'foggy':>8} {'Macro F1':>10} {'cloudy F1':>10}")
    shown = 0
    for r in stage2:
        b = {"cloudy": r['cv'], "sunny": r['sv'], "rainy": r['rv'], "foggy": r['fv']}
        if r['rv'] == 0.0 and r['fv'] == 0.0:
            continue
        shown += 1
        print(f"{shown:<5} {r['cv']:>8.2f} {r['sv']:>8.2f} {r['rv']:>8.2f} "
              f"{r['fv']:>8.2f} {r['mf1']:>10.4f} {r['cf1']:>10.4f}")
        if shown >= 10:
            break

    # ---- 最终结果 ----
    # 选择更优的（S2 最优 vs S1 最优）
    best_bias = best_s2_bias
    best_f1 = best_s2_f1
    best_per = best_s2_per

    print(f"\n{'='*60}")
    print(f"  ** 最优 Bias 配置 **")
    print(f"{'='*60}")
    print(f"  Macro F1:  {best_f1:.4f}  (baseline={base_f1:.4f}, Δ={best_f1-base_f1:+.4f})")
    print(f"  cloudy F1: {best_per['cloudy']:.4f}  (baseline={base_per_dict.get('cloudy',0):.4f}, Δ={best_per['cloudy']-base_per_dict.get('cloudy',0):+.4f})")
    print(f"  Bias:      {best_bias}")
    print(f"\n  各维度 F1 (baseline → best):")
    for name in class_names:
        base_v = base_per_dict.get(name, 0)
        best_v = best_per.get(name, 0)
        delta = best_v - base_v
        bar = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "─")
        print(f"    {name:<12s}: {base_v:.4f} → {best_v:.4f}  ({bar} {delta:+.4f})")
    print(f"{'='*60}")

    # 可直接粘贴的配置
    print(f"\n# 粘贴到 config.py：")
    print(f'CONFIG["logit_bias"] = {json.dumps(best_bias, indent=4)}')

    # 保存完整结果
    out_path = Path("results/bias_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "checkpoint": args.ckpt,
        "class_names": class_names,
        "formula": "logits - bias",
        "baseline": {"macro_f1": round(base_f1, 6), "per_class": {k: round(v, 6) for k, v in base_per_dict.items()}},
        "best": {"bias": best_bias, "macro_f1": round(best_f1, 6),
                  "per_class": {k: round(v, 6) for k, v in best_per.items()}},
        "stage1_top10": stage1[:10],
        "stage2_top10": stage2[:10],
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存: {out_path}")


if __name__ == "__main__":
    main()
