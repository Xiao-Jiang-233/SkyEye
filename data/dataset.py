# ============ data/dataset.py ============
"""数据集加载：ImageFolder → DataLoader，含类别权重计算"""
import hashlib
import json
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

_CACHE_SCHEMA_VERSION = 1
_MANIFEST_NAME = ".manifest.json"
_PREPARED_CACHE = {}


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


def _collect_zip_class_names(zip_path):
    """不解压 ZIP，直接从成员路径识别类别目录。"""
    member_parts = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = [
                part
                for part in member.filename.replace("\\", "/").split("/")
                if part
            ]
            if any(part.startswith((".", "_")) for part in parts):
                continue
            if len(parts) >= 2:
                member_parts.append(parts)

    if not member_parts:
        return []

    top_dirs = {parts[0] for parts in member_parts}
    if len(top_dirs) == 1 and all(len(parts) >= 3 for parts in member_parts):
        member_parts = [parts[1:] for parts in member_parts]
        top_dirs = {parts[0] for parts in member_parts}

    split_keys = {"train", "test", "val", "validation"}
    if top_dirs & split_keys:
        return sorted({
            parts[1]
            for parts in member_parts
            if len(parts) >= 3 and parts[0] in split_keys
        })
    return sorted(top_dirs)


def _collect_entry_class_names(entry):
    source_path = entry["path"]
    if os.path.isfile(source_path) and source_path.lower().endswith(".zip"):
        return _collect_zip_class_names(source_path)

    src_dir = _resolve_src_dir(entry)
    if src_dir is None:
        return []
    return _collect_class_names(src_dir)


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
        try:
            for entry in entries:
                # 收集所有源类名（支持嵌套型 train/test/{class}/）
                src_classes = _collect_entry_class_names(entry)
                auto_map = _auto_class_map(src_classes, target, aliases)
                if auto_map:
                    entry["class_map"] = auto_map
        except Exception:
            _cleanup_resolved_sources(entries)
            raise

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
    resolved_path = entry.get("_resolved_path")
    if resolved_path and os.path.isdir(resolved_path):
        return resolved_path

    src_path = entry["path"]

    if not os.path.exists(src_path):
        print(f"  ⚠ Source not found: {src_path}")
        return None

    # 如果是 zip 文件，解压到临时目录
    if os.path.isfile(src_path) and src_path.lower().endswith('.zip'):
        tmp_dir = tempfile.mkdtemp(prefix="skyeye_data_")
        print(f"  [ZIP] Extracting: {src_path} -> {tmp_dir}")
        try:
            with zipfile.ZipFile(src_path, 'r') as zf:
                temp_root = os.path.abspath(tmp_dir)
                for member in zf.infolist():
                    destination = os.path.abspath(
                        os.path.join(tmp_dir, member.filename)
                    )
                    if os.path.commonpath([temp_root, destination]) != temp_root:
                        raise ValueError(
                            f"ZIP 包含越界路径，拒绝解压: {member.filename}"
                        )
                zf.extractall(tmp_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        # 如果 zip 内只有一层子目录，自动进入（去掉外层壳）
        members = [f for f in os.listdir(tmp_dir)
                   if os.path.isdir(os.path.join(tmp_dir, f)) and not f.startswith('_')]
        if len(members) == 1 and not any(
                os.path.isfile(os.path.join(tmp_dir, f)) for f in os.listdir(tmp_dir)
        ):
            resolved_path = os.path.join(tmp_dir, members[0])
        else:
            resolved_path = tmp_dir
        entry["_temp_root"] = tmp_dir
        entry["_resolved_path"] = resolved_path
        return resolved_path

    # 普通目录
    if os.path.isdir(src_path):
        return src_path

    print(f"  ⚠ Source is not a directory or zip: {src_path}")
    return None


def _cleanup_resolved_sources(entries):
    """清理本轮自动解压生成的 ZIP 临时目录。"""
    for entry in entries:
        temp_root = entry.pop("_temp_root", None)
        entry.pop("_resolved_path", None)
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


def _iter_class_dirs(src_dir, class_map, target_classes, skip_classes=None):
    """
    遍历数据源目录，生成 (src_label, file_paths, dst_class_name)

    自动识别两种目录结构：
    - 平铺型：src/cloudy/*.jpg, src/haze/*.jpg ...
    - 嵌套型：src/train/cloudy/*.jpg, src/test/cloudy/*.jpg ...  → 自动合并 train+test

    Args:
        src_dir: str — 数据源根目录
        class_map: dict — {src_name: dst_name} 类名映射
        target_classes: list[str] — 目标类别列表
        skip_classes: set | None — 暂时跳过的类别（即使映射到 target_classes 也跳过）

    Yields:
        (label: str, files: list[str], dst_class_name: str)
    """
    skip = set(skip_classes or [])
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
                if dst_cls in skip:
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
            if dst_cls in skip:
                print(f"    ⏭ {src_cls}→{dst_cls}: in skip_classes, skipping")
                continue
            yield (src_cls, files, dst_cls)


def _canonical_json(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_signature(entry):
    """生成数据源指纹，检测源目录或 ZIP 是否发生变化。"""
    source_path = os.path.abspath(entry["path"])
    signature = {
        "path": source_path,
        "class_map": entry.get("class_map", {}),
    }

    if not os.path.exists(source_path):
        signature["kind"] = "missing"
        return signature

    stat = os.stat(source_path)
    signature["mtime_ns"] = stat.st_mtime_ns
    if os.path.isfile(source_path):
        signature.update({"kind": "file", "size": stat.st_size})
        return signature

    signature["kind"] = "directory"
    directories = []
    total_files = 0
    total_size = 0
    metadata_digest = hashlib.sha256()
    for current, dirnames, filenames in os.walk(source_path):
        dirnames.sort()
        filenames.sort()
        current_stat = os.stat(current)
        relative_dir = os.path.relpath(current, source_path).replace("\\", "/")
        directories.append({
            "path": relative_dir,
            "mtime_ns": current_stat.st_mtime_ns,
            "file_count": len(filenames),
        })
        total_files += len(filenames)
        for filename in filenames:
            file_path = os.path.join(current, filename)
            file_stat = os.stat(file_path)
            total_size += file_stat.st_size
            relative_file = (
                filename
                if relative_dir == "."
                else f"{relative_dir}/{filename}"
            )
            metadata_digest.update(
                f"{relative_file}\0{file_stat.st_size}\0"
                f"{file_stat.st_mtime_ns}\n".encode("utf-8")
            )
    signature["total_files"] = total_files
    signature["total_size"] = total_size
    signature["metadata_sha256"] = metadata_digest.hexdigest()
    signature["directories"] = directories
    return signature


def _build_manifest_spec(entries):
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "config": {
            "class_names": list(CONFIG.get("class_names", [])),
            "active_class_names": list(CONFIG.get("active_class_names", [])),
            "skip_classes": list(CONFIG.get("skip_classes", [])),
            "class_aliases": dict(CONFIG.get("class_aliases", {})),
        },
        "sources": [_source_signature(entry) for entry in entries],
    }
    payload["fingerprint"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _read_manifest(root):
    path = os.path.join(root, _MANIFEST_NAME)
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_manifest(root, manifest):
    path = os.path.join(root, _MANIFEST_NAME)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{_MANIFEST_NAME}.",
        suffix=".tmp",
        dir=root,
    )
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _validate_dataset_contract(root):
    """校验 ImageFolder 类别索引与配置完全一致。"""
    if not os.path.isdir(root):
        raise RuntimeError(f"数据目录不存在: {root}")

    dataset = ImageFolder(root)
    expected_classes = list(CONFIG.get("active_class_names", []))
    actual_classes = list(dataset.classes)
    if actual_classes != expected_classes:
        raise RuntimeError(
            "数据类别索引与配置不一致: "
            f"ImageFolder={actual_classes}, CONFIG={expected_classes}"
        )

    counts = np.bincount(dataset.targets, minlength=len(actual_classes))
    empty_classes = [
        class_name
        for class_name, count in zip(actual_classes, counts)
        if int(count) == 0
    ]
    if empty_classes:
        raise RuntimeError(f"以下类别没有可用图片: {empty_classes}")

    return {
        class_name: int(count)
        for class_name, count in zip(actual_classes, counts)
    }


def _merge_sources(destination, entries):
    target_classes = CONFIG.get("class_names", [])
    skip_classes = CONFIG.get("skip_classes", [])
    total_files = 0

    for i, entry in enumerate(entries):
        src_dir = _resolve_src_dir(entry)
        if src_dir is None:
            continue

        class_map = entry["class_map"]
        label = os.path.basename(entry["path"])
        print(f"[{i + 1}/{len(entries)}] Merging: {label} -> {destination}")
        if class_map:
            print(f"       class_map: {class_map}")

        for src_label, files, dst_class_name in _iter_class_dirs(
                src_dir, class_map, target_classes, skip_classes,
        ):
            dst_class_dir = os.path.join(destination, dst_class_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            copied = 0
            for src_file in files:
                dst_file = os.path.join(
                    dst_class_dir,
                    os.path.basename(src_file),
                )
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied += 1

            if copied:
                arrow = (
                    f"->{dst_class_name}"
                    if not src_label.endswith(dst_class_name)
                    else ""
                )
                print(f"    {src_label} {arrow}: {copied} files".replace("  ", " "))
                total_files += copied

    existing_classes = sorted(
        name
        for name in os.listdir(destination)
        if os.path.isdir(os.path.join(destination, name))
    )
    print(
        f"Dataset merge complete - {total_files} total files "
        f"across {len(existing_classes)} classes: {existing_classes}"
    )


def _source_inventory(entries):
    """计算按现有合并规则应出现的逐类文件名集合。"""
    target_classes = CONFIG.get("class_names", [])
    skip_classes = CONFIG.get("skip_classes", [])
    inventory = {
        class_name: set()
        for class_name in CONFIG.get("active_class_names", [])
    }

    for entry in entries:
        src_dir = _resolve_src_dir(entry)
        if src_dir is None:
            continue
        for _, files, dst_class_name in _iter_class_dirs(
                src_dir,
                entry["class_map"],
                target_classes,
                skip_classes,
        ):
            inventory.setdefault(dst_class_name, set()).update(
                os.path.basename(path) for path in files
            )
    return inventory


def _cache_inventory(root):
    inventory = {}
    for class_name in CONFIG.get("active_class_names", []):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            return None
        inventory[class_name] = {
            filename
            for filename in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, filename))
        }
    return inventory


def _adopt_legacy_cache(root, entries, manifest_spec):
    """旧缓存与当前源完全匹配时，仅补写清单，避免无意义全量复制。"""
    try:
        counts = _validate_dataset_contract(root)
        if _cache_inventory(root) != _source_inventory(entries):
            return False
    except (RuntimeError, FileNotFoundError, OSError):
        return False

    manifest = dict(manifest_spec)
    manifest["class_counts"] = counts
    _write_manifest(root, manifest)
    print("[数据缓存] 旧缓存校验通过，已迁移为清单缓存")
    return True


def _replace_cache_directory(build_root, destination):
    """切换已验证缓存，失败时恢复原缓存。"""
    backup = None
    if os.path.exists(destination):
        backup = f"{destination}.backup-{os.getpid()}"
        suffix = 0
        while os.path.exists(backup):
            suffix += 1
            backup = f"{destination}.backup-{os.getpid()}-{suffix}"
        os.replace(destination, backup)

    try:
        os.replace(build_root, destination)
    except Exception:
        if backup is not None and not os.path.exists(destination):
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as error:
                print(
                    f"[警告] 旧数据缓存清理失败，可稍后删除 {backup}: {error}"
                )


def prepare_data():
    """
    将多个数据源合并到带清单校验的可写缓存目录

    自动处理：
    - 多源合并（逐类逐文件复制，同名跳过）
    - 平铺型 & 嵌套型（train/test/val/{class}/）目录结构
    - .zip 文件自动解压到临时目录
    - 类名映射（class_map + class_aliases）
    - 缺失数据源跳过并警告
    - 数据源或类别配置变化时原子重建缓存

    Returns:
        str: 合并后可写数据目录路径
    """
    dst = os.path.abspath(CONFIG["writable_root"])
    config_key = _canonical_json({
        "destination": dst,
        "data_roots": CONFIG.get("data_roots", CONFIG.get("data_root")),
        "class_names": CONFIG.get("class_names", []),
        "active_class_names": CONFIG.get("active_class_names", []),
        "skip_classes": CONFIG.get("skip_classes", []),
        "class_aliases": CONFIG.get("class_aliases", {}),
    })
    if _PREPARED_CACHE.get(dst) == config_key and os.path.isdir(dst):
        return dst

    entries = _get_data_roots()
    build_root = None
    try:
        expected_manifest = _build_manifest_spec(entries)
        current_manifest = _read_manifest(dst)
        if (
            current_manifest is not None
            and current_manifest.get("fingerprint")
            == expected_manifest["fingerprint"]
        ):
            try:
                counts = _validate_dataset_contract(dst)
            except (RuntimeError, FileNotFoundError) as error:
                print(f"[数据缓存] 校验失败，将重新构建: {error}")
            else:
                if counts == current_manifest.get("class_counts"):
                    _PREPARED_CACHE[dst] = config_key
                    return dst
                print("[数据缓存] 类别数量变化，将重新构建")
        elif os.path.exists(dst):
            if (
                current_manifest is None
                and _adopt_legacy_cache(dst, entries, expected_manifest)
            ):
                _PREPARED_CACHE[dst] = config_key
                return dst
            reason = (
                "缺少缓存清单"
                if current_manifest is None
                else "配置或数据源发生变化"
            )
            print(f"[数据缓存] {reason}，将重新构建")

        parent = os.path.dirname(dst)
        os.makedirs(parent, exist_ok=True)
        build_root = tempfile.mkdtemp(
            prefix=f".{os.path.basename(dst)}.build-",
            dir=parent,
        )
        _merge_sources(build_root, entries)
        counts = _validate_dataset_contract(build_root)

        manifest = dict(expected_manifest)
        manifest["class_counts"] = counts
        _write_manifest(build_root, manifest)
        _replace_cache_directory(build_root, dst)
        build_root = None

        _PREPARED_CACHE[dst] = config_key
        return dst
    finally:
        if build_root is not None and os.path.exists(build_root):
            shutil.rmtree(build_root, ignore_errors=True)
        _cleanup_resolved_sources(entries)


def create_dataloaders(data_root=None, img_size=None, batch_size=None, num_workers=None,
                       cloudy_oversample=False, sunny_oversample=False):
    """
    创建训练和验证 DataLoader

    Args:
        data_root: str — 数据根目录（默认从 CONFIG 读取）
        img_size: int — 图片尺寸
        batch_size: int — 批次大小
        num_workers: int — 数据加载线程数
        cloudy_oversample: bool — 是否启用 Cloudy 过采样 2×（DRW 时仅在后期开启）

    Returns:
        tuple: (train_loader, val_loader, class_counts, class_names)
    """
    cfg = CONFIG
    root = data_root if data_root is not None else prepare_data()
    size = img_size if img_size is not None else cfg["img_size"]
    bs = batch_size if batch_size is not None else cfg["batch_size"]
    nw = num_workers if num_workers is not None else cfg["num_workers"]

    # 全量加载以获取类别分布
    full_dataset = ImageFolder(root)
    expected_classes = list(cfg.get("active_class_names", []))
    if full_dataset.classes != expected_classes:
        raise RuntimeError(
            "ImageFolder 类别索引与模型配置不一致: "
            f"dataset={full_dataset.classes}, CONFIG={expected_classes}"
        )
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

    # ① Cloudy 过采样 2×（DRW: 仅在后 40% epoch 启用，由 train_teacher 控制）
    class_to_idx_map = full_dataset.class_to_idx
    if cloudy_oversample and "cloudy" in class_to_idx_map:
        cloudy_label = class_to_idx_map["cloudy"]
        cloudy_train_idx = [i for i in train_idx if full_dataset.targets[i] == cloudy_label]
        oversample_count = len(cloudy_train_idx)  # 1× 追加 → 总共 2×
        train_idx = np.concatenate([train_idx, cloudy_train_idx])
        print(f"Cloudy oversampling: {oversample_count} → {oversample_count * 2} "
              f"({oversample_count} duplicated, 2×)")
    elif "cloudy" in class_to_idx_map:
        cloudy_label = class_to_idx_map["cloudy"]
        cloudy_count = sum(1 for i in train_idx if full_dataset.targets[i] == cloudy_label)
        print(f"Cloudy oversampling: OFF ({cloudy_count} original samples)")

    # ② Sunny 过采样 2×（方案 C：对抗 sunny→cloudy 非对称偏向）
    if sunny_oversample and "sunny" in class_to_idx_map:
        sunny_label = class_to_idx_map["sunny"]
        sunny_train_idx = [i for i in train_idx if full_dataset.targets[i] == sunny_label]
        sunny_count = len(sunny_train_idx)
        train_idx = np.concatenate([train_idx, sunny_train_idx])
        print(f"Sunny oversampling: {sunny_count} → {sunny_count * 2} "
              f"({sunny_count} duplicated, 2×)")
    elif "sunny" in class_to_idx_map:
        sunny_label = class_to_idx_map["sunny"]
        sunny_count = sum(1 for i in train_idx if full_dataset.targets[i] == sunny_label)
        print(f"Sunny oversampling: OFF ({sunny_count} original samples)")

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
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
    )

    class_names = full_dataset.classes
    if not cloudy_oversample:
        print(f"Classes: {class_names}")
        print(f"Class distribution: {dict(zip(class_names, class_counts.astype(int)))}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    return train_loader, val_loader, class_counts, class_names


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
