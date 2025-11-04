#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

SPLIT_NAMES = ("train", "valid", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def classify_line(tokens: List[str]) -> str:
    """
    Returns one of: 'bbox', 'seg', 'empty', 'malformed'
    """
    if not tokens:
        return 'empty'
    if not all(is_number(t) for t in tokens):
        return 'malformed'
    n = len(tokens)
    if n == 5:
        return 'bbox'
    if n >= 7 and (n % 2 == 1):
        return 'seg'
    return 'malformed'

def scan_label_file(path: Path) -> Tuple[Dict[str, int], bool]:
    """
    Scans a single .txt label file and returns:
      - counts dict
      - mixed_file flag (True if both bbox and seg are present in the file)
    """
    counts = {'bbox': 0, 'seg': 0, 'empty': 0, 'malformed': 0}
    types_seen = set()
    try:
        with path.open('r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    counts['empty'] += 1
                    continue
                tokens = line.split()
                kind = classify_line(tokens)
                counts[kind] += 1
                if kind in ('bbox', 'seg'):
                    types_seen.add(kind)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
        counts['malformed'] += 1
    mixed_file = (len(types_seen) > 1)
    return counts, mixed_file

def find_image_for_label(label_file: Path, images_dir: Path) -> Optional[Path]:
    """
    Given a label file .../labels/X.txt, try to find the corresponding image .../images/X.<ext>
    """
    stem = label_file.stem
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None

def list_background_images(images_dir: Path, labels_dir: Path) -> List[Path]:
    """
    Return images in images_dir that do not have a matching .txt in labels_dir.
    Matching rule: same stem, label path = labels_dir / "<stem>.txt"
    """
    background = []
    if not images_dir.is_dir():
        return background
    label_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else set()
    for img in images_dir.iterdir():
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            if img.stem not in label_stems:
                background.append(img)
    return background

def scan_split(split_dir: Path) -> Optional[Dict[str, int]]:
    """
    Scan one split for stats. Returns None if there is no labels folder.
    """
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"

    if not labels_dir.is_dir():
        return None

    split_stats = {
        'files_found': 0,
        'scanned_files': 0,
        'bbox': 0,
        'seg': 0,
        'empty': 0,
        'malformed': 0,
        'mixed_files': 0,
        'background_images': 0
    }

    txt_files = sorted(labels_dir.glob("*.txt"))
    split_stats['files_found'] = len(txt_files)

    # Stats over label files
    for lf in txt_files:
        counts, mixed = scan_label_file(lf)
        split_stats['scanned_files'] += 1
        split_stats['bbox'] += counts['bbox']
        split_stats['seg'] += counts['seg']
        split_stats['empty'] += counts['empty']
        split_stats['malformed'] += counts['malformed']
        if mixed:
            split_stats['mixed_files'] += 1

    # Background images
    split_stats['background_images'] = len(list_background_images(images_dir, labels_dir))

    return split_stats

def classify_label_file_type(label_file: Path) -> str:
    """
    Return one of: 'bbox_only', 'seg_only', 'mixed', 'empty_only', 'malformed_only', 'other'
    Used for deletion selection.
    """
    counts, mixed = scan_label_file(label_file)
    if mixed:
        return 'mixed'
    if counts['bbox'] > 0 and counts['seg'] == 0 and counts['malformed'] == 0:
        return 'bbox_only'
    if counts['seg'] > 0 and counts['bbox'] == 0 and counts['malformed'] == 0:
        return 'seg_only'
    if counts['empty'] > 0 and counts['bbox'] == 0 and counts['seg'] == 0 and counts['malformed'] == 0:
        return 'empty_only'
    if counts['malformed'] > 0 and counts['bbox'] == 0 and counts['seg'] == 0:
        return 'malformed_only'
    return 'other'

def delete_targets_in_split(split_dir: Path,
                            remove_type: str,
                            delete_mixed: bool,
                            apply: bool,
                            list_actions: bool) -> Tuple[int, int]:
    """
    For a given split, delete label files (and their images) that match the requested remove_type:
      - remove_type in {'bbox','seg'}
      - delete_mixed: if True, also delete mixed files (contains both types)
    Returns (labels_deleted, images_deleted)
    """
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"
    if not labels_dir.is_dir():
        return (0, 0)

    labels_deleted = 0
    images_deleted = 0

    for label_file in sorted(labels_dir.glob("*.txt")):
        file_type = classify_label_file_type(label_file)

        should_delete = False
        if remove_type == 'bbox' and file_type == 'bbox_only':
            should_delete = True
        elif remove_type == 'seg' and file_type == 'seg_only':
            should_delete = True
        elif delete_mixed and file_type == 'mixed':
            # If user asked to delete mixed too, we consider mixed as matching both remove types
            should_delete = True

        if not should_delete:
            continue

        img_path = find_image_for_label(label_file, images_dir)

        if list_actions:
            print(f"[DELETE] Label: {label_file}")
            if img_path:
                print(f"[DELETE] Image: {img_path}")
            else:
                print(f"[WARN] No matching image found for {label_file.stem} in {images_dir}")

        if apply:
            # Delete label
            try:
                label_file.unlink()
                labels_deleted += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete label {label_file}: {e}", file=sys.stderr)
            # Delete image if exists
            if img_path and img_path.exists():
                try:
                    img_path.unlink()
                    images_deleted += 1
                except Exception as e:
                    print(f"[ERROR] Failed to delete image {img_path}: {e}", file=sys.stderr)

    return (labels_deleted, images_deleted)

def print_split_report(split_name: str, stats: Optional[Dict[str, int]]):
    print(f"\n=== Split: {split_name} ===")
    if stats is None:
        print("labels/ folder not found. Skipping.")
        return
    print(f"Label files scanned   : {stats['scanned_files']} (out of {stats['files_found']} found)")
    print(f"Bounding boxes (lines): {stats['bbox']}")
    print(f"Segmentations (lines) : {stats['seg']}")
    print(f"Empty lines           : {stats['empty']}")
    print(f"Malformed lines       : {stats['malformed']}")
    print(f"Background images     : {stats['background_images']}")
    if stats['mixed_files'] > 0:
        print(f"Files mixing types    : {stats['mixed_files']}  <-- ⚠️ mixed types inside some label files")

    # Split-level type assessment
    if stats['bbox'] > 0 and stats['seg'] == 0:
        print("Split type            : Object Detection (bbox-only)")
    elif stats['seg'] > 0 and stats['bbox'] == 0:
        print("Split type            : Instance Segmentation (polygon-only)")
    elif stats['bbox'] == 0 and stats['seg'] == 0:
        print("Split type            : No annotations detected")
    else:
        print("Split type            : ⚠️ MIXED (both bbox and segmentation present)")

def parse_splits_arg(arg: Optional[str]) -> Tuple[str, ...]:
    if not arg:
        return SPLIT_NAMES
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    # preserve user order, but filter to known names
    valid = tuple(p for p in parts if p in SPLIT_NAMES)
    if not valid:
        print(f"[WARN] --splits provided but none matched {SPLIT_NAMES}. Using all.", file=sys.stderr)
        return SPLIT_NAMES
    return valid

def main():
    ap = argparse.ArgumentParser(
        description="Scan a YOLO dataset (train/valid|val/test splits) to count bbox vs segmentation, detect mixed types, detect background images, and optionally remove files by type."
    )
    ap.add_argument("dataset_root", type=str, help="Path to the dataset folder containing splits (e.g., /path/to/dataset)")
    ap.add_argument("--splits", type=str, default=None,
                    help="Comma-separated subset of splits to process (choices: train,valid,val,test). Default: all found.")
    ap.add_argument("--list-background", action="store_true",
                    help="List background image file paths (images without matching label).")
    ap.add_argument("--remove-type", choices=["bbox", "seg"], default=None,
                    help="If set, prepare deletion of label+image files of this type (bbox-only or seg-only).")
    ap.add_argument("--delete-mixed", action="store_true",
                    help="When used with --remove-type, also delete mixed label files (contain both types).")
    ap.add_argument("--apply", action="store_true",
                    help="Actually perform deletions. Without this, it only prints what would be deleted (dry run).")
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation when using --apply.")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Dataset path invalid: {root}", file=sys.stderr)
        sys.exit(2)

    selected_splits = parse_splits_arg(args.splits)

    # Discover splits under root (keep standard names order)
    found = []
    for name in selected_splits:
        p = root / name
        if p.is_dir():
            found.append((name, p))
    if not found:
        print(f"[ERROR] No requested splits found under {root}. Looked for: {selected_splits}", file=sys.stderr)
        sys.exit(1)

    # Scan mode always runs; deletion is optional
    overall = {
        'bbox': 0, 'seg': 0, 'empty': 0, 'malformed': 0,
        'mixed_files': 0, 'splits_mixed': 0, 'splits_total': 0,
        'background_images': 0
    }

    # First pass: reporting
    for split_name, split_dir in found:
        stats = scan_split(split_dir)
        print_split_report(split_name, stats)
        if stats is None:
            continue
        overall['splits_total'] += 1
        for k in ('bbox', 'seg', 'empty', 'malformed', 'mixed_files', 'background_images'):
            overall[k] += stats[k]
        if stats['bbox'] > 0 and stats['seg'] > 0:
            overall['splits_mixed'] += 1

        # Optionally list background images
        if args.list_background:
            labels_dir = split_dir / "labels"
            images_dir = split_dir / "images"
            bgs = list_background_images(images_dir, labels_dir)
            if bgs:
                print("  Background image files:")
                for p in bgs:
                    print(f"    - {p}")
            else:
                print("  No background images found.")

    # Overall summary
    print("\n=== Overall Summary ===")
    print(f"Total bbox lines        : {overall['bbox']}")
    print(f"Total segmentation lines: {overall['seg']}")
    print(f"Total empty lines       : {overall['empty']}")
    print(f"Total malformed lines   : {overall['malformed']}")
    print(f"Files mixing types      : {overall['mixed_files']}")
    print(f"Mixed splits            : {overall['splits_mixed']} / {overall['splits_total']}")
    print(f"Total background images : {overall['background_images']}")

    # Deletion phase (optional)
    if args.remove_type:
        if args.apply and not args.yes:
            print("[ABORT] --apply requires --yes to confirm destructive deletion.", file=sys.stderr)
            sys.exit(4)

        mode = "DRY-RUN" if not args.apply else "APPLY"
        print(f"\n=== Deletion Phase ({mode}) ===")
        print(f"Target type     : {args.remove_type} "
              f"({'and mixed files' if args.delete_mixed else 'mixed files NOT included'})")
        total_labels_deleted = 0
        total_images_deleted = 0

        for split_name, split_dir in found:
            labels_deleted, images_deleted = delete_targets_in_split(
                split_dir=split_dir,
                remove_type=args.remove_type,
                delete_mixed=args.delete_mixed,
                apply=args.apply,
                list_actions=True
            )
            print(f"[{split_name}] deleted labels: {labels_deleted}, images: {images_deleted}")
            total_labels_deleted += labels_deleted
            total_images_deleted += images_deleted

        print(f"\n=== Deletion Summary ({mode}) ===")
        print(f"Total labels deleted: {total_labels_deleted}")
        print(f"Total images deleted: {total_images_deleted}")

if __name__ == "__main__":
    main()
