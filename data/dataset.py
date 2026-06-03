# ============ data/dataset.py ============
"""数据集加载：ImageFolder → DataLoader，含类别权重计算"""
import os
import shutil
import tempfile
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import CONFIG
from data.augmentations import get_train_transforms, get_val_transforms


def _normalize_entry(entry):
    """
    将 data_roots 条目统一为 {"path": str, "class_map": dict}

    Args:
        entry: str | dict — 字符串路径或含 path + class_map 的字典

    Returns:
        dict: {"path": str, "class_map": dict}
    """
    if isinstance(entry, str):
        return {"path": entry, "class_map": {}}
    if isinstance(entry, dict):
        return {"path": entry["path"], "class_map": entry.get("class_map", {})}
    raise TypeError(f"data_roots 条目类型错误: {type(entry)}，应为 str 或 dict")


def _collect_class_names(src_dir):
    """
    从数据源目录收集所有唯一的类目录名

    自动识别平铺型和嵌套型（train/test/{class}/）结构。

    Args:
        src_dir: str — 数据源根目录

    Returns:
        list[str]: 去重后的类目录名列表
    """
    top_dirs = [d for d in os.listdir(src_dir)
                if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('_')]
    top_set = set(top_dirs)
    SPLIT_KEYS = {'train', 'test', 'val', 'validation'}

    if top_set & SPLIT_KEYS:
        # 嵌套型：从 split 子目录收集类名
        classes = set()
        for split_name in sorted(top_set & SPLIT_KEYS):
            split_dir = os.path.join(src_dir, split_name)
            for cls in os.listdir(split_dir):
                if os.path.isdir(os.path.join(split_dir, cls)):
                    classes.add(cls)
        return sorted(classes)
    else:
        # 平铺型：顶层目录即为类名
        return top_dirs


def _auto_class_map(src_class_dirs, target_classes, aliases):
    """
    自动生成类名映射：源目录名 → 标准类名

    匹配规则（按优先级）:
    1. 精确匹配 → 无需映射
    2. 命中 aliases 表 → 用别名映射
    3. 模糊匹配（小写后包含关系）→ 自动关联
    4. 以上都不匹配 → 警告并跳过

    Args:
        src_class_dirs: list[str] — 源数据集下的类目录名
        target_classes: list[str] — 目标类别名（class_names）
        aliases: dict — 别名映射表

    Returns:
        dict: {src_name: target_name}  需要重命名的映射
    """
    class_map = {}
    target_set = set(target_classes)
    target_lower = {t.lower(): t for t in target_classes}

    for src in src_class_dirs:
        # 1) 精确匹配
        if src in target_set:
            continue

        # 2) 查别名表
        if src in aliases:
            mapped = aliases[src]
            if mapped in target_set:
                class_map[src] = mapped
                continue

        # 3) 模糊匹配（小写后完全一致）
        src_lower = src.lower()
        if src_lower in target_lower:
            class_map[src] = target_lower[src_lower]
            continue

        # 4) 无法匹配
        print(f"    ⚠ Unknown class '{src}' — not in class_names or aliases, will skip")

    return class_map


def _scan_datasets_dir(datasets_root, target_classes, aliases):
    """
    自动扫描 datasets/ 目录，发现所有可用的数据源

    搜索策略:
    a) <hash>/weather_classification/  — 标准 weather 数据集目录
    b) <hash>/*.zip                    — zip 压缩包（含 weather 类别子目录）
    c) <hash>/ 直接包含类目录           — 无外层壳的情况

    Args:
        datasets_root: str — datasets/ 目录路径
        target_classes: list[str] — 目标类别
        aliases: dict — 类名别名表

    Returns:
        list[dict]: [{"path": ..., "class_map": {...}}, ...]
    """
    if not os.path.isdir(datasets_root):
        print(f"  ⚠ datasets/ directory not found: {datasets_root}")
        return []

    entries = []
    for import_name in sorted(os.listdir(datasets_root)):
        import_dir = os.path.join(datasets_root, import_name)
        if not os.path.isdir(import_dir) or import_name.startswith('.'):
            continue

        found = False

        # a) 查找 weather_classification/ 子目录
        weather_dir = os.path.join(import_dir, "weather_classification")
        if os.path.isdir(weather_dir):
            entries.append({"path": weather_dir, "class_map": {}})
            found = True

        # b) 查找 .zip 文件
        for fname in sorted(os.listdir(import_dir)):
            if fname.lower().endswith('.zip'):
                zip_path = os.path.join(import_dir, fname)
                entries.append({"path": zip_path, "class_map": {}})
                found = True

        # c) 直接包含类目录（无 weather_classification 壳）
        if not found:
            subdirs = [d for d in os.listdir(import_dir)
                       if os.path.isdir(os.path.join(import_dir, d)) and not d.startswith('_')]
            # 检查是否有子目录名匹配我们的类别（精确或别名）
            matched = sum(1 for d in subdirs
                          if d in target_classes or d in aliases or d.lower() in [t.lower() for t in target_classes])
            if matched >= 2:  # 至少匹配2个类别才认为是合法数据源
                entries.append({"path": import_dir, "class_map": {}})
                found = True

        if not found:
            print(f"  ⚠ Skipping {import_name}: no weather_classification/, zip, or class dirs found")

    return entries


def _get_data_roots():
    """
    获取所有数据源（归一化后），支持 auto / 列表 / 向后兼容

    - "auto": 自动扫描 datasets/ 发现所有导入
    - list:  手动指定的数据源列表
    - 回退:  读取旧版 data_root 单路径

    Returns:
        list[dict]: [{"path": ..., "class_map": {...}}, ...]
    """
    cfg = CONFIG
    raw = cfg.get("data_roots", [])

    # --- auto 模式：自动发现 ---
    if raw == "auto":
        target = cfg.get("class_names", [])
        aliases = cfg.get("class_aliases", {})
        datasets_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
        )
        print("=" * 50)
        print("Auto-discovering datasets in datasets/ ...")
        print(f"  Target classes: {target}")
        print(f"  Aliases: {aliases}")
        print("=" * 50)

        entries = _scan_datasets_dir(datasets_root, target, aliases)

        if not entries:
            raise ValueError(
                "auto 模式未发现任何数据集！请确认 datasets/ 目录下有导入的数据。\n"
                "或手动指定 data_roots 列表。"
            )

        # 对每个自动发现的条目，自动生成 class_map
        for entry in entries:
            # 先解析目录（可能需解压 zip）
            src_dir = _resolve_src_dir(entry)
            if src_dir is None:
                continue
            # 收集所有源类名（支持嵌套型 train/test/{class}/）
            src_classes = _collect_class_names(src_dir)
            auto_map = _auto_class_map(src_classes, target, aliases)
            if auto_map:
                entry["class_map"] = auto_map

        print(f"Auto-discovered {len(entries)} data source(s)\n")
        return entries

    # --- 列表模式：手动指定 ---
    if isinstance(raw, list) and raw:
        return [_normalize_entry(e) for e in raw]

    # --- 向后兼容：旧版 data_root ---
    if "data_root" in cfg:
        return [_normalize_entry(cfg["data_root"])]

    raise ValueError("config 中缺少 data_roots 或 data_root，请至少配置一个数据源路径")


def _resolve_src_dir(entry):
    """
    解析数据源的实际目录路径

    - 目录直接返回
    - .zip 文件自动解压到临时目录，返回解压后的路径

    Args:
        entry: dict — {"path": ..., "class_map": ...}

    Returns:
        str | None: 可遍历的目录路径，不存在则返回 None
    """
    src_path = entry["path"]

    if not os.path.exists(src_path):
        print(f"  ⚠ Source not found: {src_path}")
        return None

    # 如果是 zip 文件，解压到临时目录
    if os.path.isfile(src_path) and src_path.lower().endswith('.zip'):
        tmp_dir = tempfile.mkdtemp(prefix="skyeye_data_")
        print(f"  [ZIP] Extracting: {src_path} -> {tmp_dir}")
        with zipfile.ZipFile(src_path, 'r') as zf:
            zf.extractall(tmp_dir)
        # 如果 zip 内只有一层子目录，自动进入（去掉外层壳）
        members = [f for f in os.listdir(tmp_dir)
                   if os.path.isdir(os.path.join(tmp_dir, f)) and not f.startswith('_')]
        if len(members) == 1 and not any(
                os.path.isfile(os.path.join(tmp_dir, f)) for f in os.listdir(tmp_dir)
        ):
            return os.path.join(tmp_dir, members[0])
        return tmp_dir

    # 普通目录
    if os.path.isdir(src_path):
        return src_path

    print(f"  ⚠ Source is not a directory or zip: {src_path}")
    return None


def _iter_class_dirs(src_dir, class_map, target_classes):
    """
    遍历数据源目录，生成 (src_label, file_paths, dst_class_name)

    自动识别两种目录结构：
    - 平铺型：src/cloudy/*.jpg, src/haze/*.jpg ...
    - 嵌套型：src/train/cloudy/*.jpg, src/test/cloudy/*.jpg ...  → 自动合并 train+test

    Args:
        src_dir: str — 数据源根目录
        class_map: dict — {src_name: dst_name} 类名映射
        target_classes: list[str] — 目标类别列表

    Yields:
        (label: str, files: list[str], dst_class_name: str)
    """
    top_dirs = [d for d in os.listdir(src_dir)
                if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('_')]
    top_dir_set = set(top_dirs)

    # 判断是否为嵌套型（含 train/test/val 子目录）
    SPLIT_KEYS = {'train', 'test', 'val', 'validation'}
    if top_dir_set & SPLIT_KEYS:
        # 嵌套型：从各 split 中收集类别文件，合并同名类
        class_files = {}  # {dst_class_name: [(label, file_path)]}
        for split_name in sorted(top_dir_set & SPLIT_KEYS):
            split_dir = os.path.join(src_dir, split_name)
            for src_cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, src_cls)
                if not os.path.isdir(cls_dir):
                    continue
                dst_cls = class_map.get(src_cls, src_cls)
                if target_classes and dst_cls not in target_classes:
                    continue
                files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                         if os.path.isfile(os.path.join(cls_dir, f))]
                label = f"{split_name}/{src_cls}"
                class_files.setdefault(dst_cls, []).append((label, files))
        # 输出合并后的每类
        for dst_cls, batches in sorted(class_files.items()):
            all_files = []
            labels = []
            for lbl, flist in batches:
                all_files.extend(flist)
                labels.append(f"{lbl}({len(flist)})")
            yield (", ".join(labels), all_files, dst_cls)
    else:
        # 平铺型：顶层目录即为类目录
        for src_cls in sorted(top_dirs):
            cls_dir = os.path.join(src_dir, src_cls)
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if os.path.isfile(os.path.join(cls_dir, f))]
            dst_cls = class_map.get(src_cls, src_cls)
            if target_classes and dst_cls not in target_classes:
                print(f"    ⚠ {src_cls}→{dst_cls}: not in class_names, skipping")
                continue
            yield (src_cls, files, dst_cls)


def prepare_data():
    """
    将多个数据源合并复制到可写目录（仅首次运行）

    自动处理：
    - 多源合并（逐类逐文件复制，同名跳过）
    - 平铺型 & 嵌套型（train/test/val/{class}/）目录结构
    - .zip 文件自动解压到临时目录
    - 类名映射（class_map + class_aliases）
    - 缺失数据源跳过并警告

    Returns:
        str: 合并后可写数据目录路径
    """
    entries = _get_data_roots()
    dst = CONFIG["writable_root"]
    target_classes = CONFIG.get("class_names", [])

    if os.path.exists(dst):
        print(f"Dataset already exists at {dst}")
        return dst

    os.makedirs(dst, exist_ok=True)
    total_files = 0

    for i, entry in enumerate(entries):
        src_dir = _resolve_src_dir(entry)
        if src_dir is None:
            continue

        class_map = entry["class_map"]
        label = os.path.basename(entry["path"])
        print(f"[{i+1}/{len(entries)}] Merging: {label} → {dst}")
        if class_map:
            print(f"       class_map: {class_map}")

        for src_label, files, dst_class_name in _iter_class_dirs(
                src_dir, class_map, target_classes,
        ):
            dst_class_dir = os.path.join(dst, dst_class_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            copied = 0
            for src_file in files:
                dst_file = os.path.join(dst_class_dir, os.path.basename(src_file))
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied += 1

            if copied:
                arrow = f"→{dst_class_name}" if not src_label.endswith(dst_class_name) else ""
                print(f"    {src_label} {arrow}: {copied} files".replace("  ", " "))
                total_files += copied

    existing_classes = [d for d in os.listdir(dst) if os.path.isdir(os.path.join(dst, d))]
    print(f"Dataset merge complete — {total_files} total files across {len(existing_classes)} classes: {existing_classes}")
    return dst


def create_dataloaders(data_root=None, img_size=None, batch_size=None, num_workers=None):
    """
    创建训练和验证 DataLoader

    Args:
        data_root: str — 数据根目录（默认从 CONFIG 读取）
        img_size: int — 图片尺寸
        batch_size: int — 批次大小
        num_workers: int — 数据加载线程数

    Returns:
        tuple: (train_loader, val_loader, class_counts)
    """
    cfg = CONFIG
    root = data_root or prepare_data()
    size = img_size or cfg["img_size"]
    bs = batch_size or cfg["batch_size"]
    nw = num_workers or cfg["num_workers"]

    # 全量加载以获取类别分布
    full_dataset = ImageFolder(root)
    num_classes = len(full_dataset.classes)
    class_counts = np.bincount(full_dataset.targets, minlength=num_classes)

    # 分层划分训练集/验证集
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg["val_split"],
        stratify=full_dataset.targets,
        random_state=cfg["seed"],
    )

    # 分别创建两个 ImageFolder 实例（不同 transform）+ Subset
    train_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_train_transforms(size)),
        train_idx,
    )
    val_ds = torch.utils.data.Subset(
        ImageFolder(root, transform=get_val_transforms(size)),
        val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
    )

    print(f"Classes: {full_dataset.classes}")
    print(f"Class distribution: {dict(zip(full_dataset.classes, class_counts.astype(int)))}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_loader, val_loader, class_counts


def compute_class_weights(class_counts):
    """
    根据类别样本数计算 FocalLoss 的 alpha 参数

    Args:
        class_counts: np.ndarray — 各类别样本数

    Returns:
        torch.Tensor: 归一化的 alpha 权重
    """
    alpha = 1.0 / (class_counts + 1e-8)
    alpha = alpha / alpha.sum() * len(class_counts)
    return torch.tensor(alpha, dtype=torch.float32)
