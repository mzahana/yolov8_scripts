#!/usr/bin/env python3
"""
Extract YOLO images by class IDs (from all splits) with progress bar.

Default behavior:
  - Copies each matching image AND its FULL, ORIGINAL label (all classes kept)
    into a per-class folder: <out_root>/class_<id>/{images,labels}/
  - If an image contains multiple requested classes, it appears in each relevant
    class_<id> directory with the same full label content preserved.

Optional:
  --label-mode filtered   # write labels filtered to the target class lines only

Usage:
    python extract_yolo_by_class.py /path/to/dataset --class-ids 0 2 5
    python extract_yolo_by_class.py /data/yolo --class-ids 1 --label-mode filtered
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Set
from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def validate_dataset_structure(dataset_path: Path) -> List[str]:
    """Validate YOLO dataset structure and return available splits."""
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    if not (dataset_path / "data.yaml").exists():
        raise ValueError(f"data.yaml not found in {dataset_path}")

    splits = []
    for split in ["train", "valid", "test"]:
        split_path = dataset_path / split
        if split_path.exists():
            if not (split_path / "images").exists() or not (split_path / "labels").exists():
                raise ValueError(f"Split '{split}' missing 'images' or 'labels' folder.")
            splits.append(split)

    if not {"train", "valid"}.issubset(set(splits)):
        raise ValueError("Dataset must include at least 'train' and 'valid' splits.")

    return splits


def iter_image_label_pairs(split_path: Path):
    """Yield (image_path, label_path) pairs for a split."""
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    for img in images_dir.iterdir():
        if img.suffix.lower() in IMAGE_EXTS:
            lbl = labels_dir / f"{img.stem}.txt"
            if lbl.exists():
                yield img, lbl


def parse_label_classes(label_path: Path):
    """Parse YOLO label: returns (parsed_pairs, raw_lines).
       parsed_pairs: list[(class_id:int, line:str)], raw_lines: list[str] (original)"""
    parsed = []
    raw_lines = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n")
            if not s.strip():
                continue
            raw_lines.append(s)
            parts = s.split()
            try:
                cid = int(parts[0])
                parsed.append((cid, s))
            except (ValueError, IndexError):
                # Keep malformed lines inside raw_lines to preserve original content,
                # but skip them from parsed class check.
                pass
    return parsed, raw_lines


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_one(src: Path, dst: Path, mode: str):
    """Copy/link file according to mode."""
    ensure_dirs(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def write_label(dst_label: Path, lines):
    """Write label lines to destination."""
    ensure_dirs(dst_label.parent)
    with dst_label.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l.rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser(description="Extract YOLO images by class IDs from all splits.")
    ap.add_argument("dataset_path", type=str, help="Path to YOLO dataset root")
    ap.add_argument("--class-ids", "-c", type=int, nargs="+", required=True,
                    help="Target class IDs to extract (e.g., 0 2 5)")
    ap.add_argument("--output-dir", "-o", type=str, default=None,
                    help="Output directory (default: <dataset_root>_by_class)")
    ap.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="How to place images in per-class dirs (default: copy)")
    ap.add_argument("--label-mode", choices=["full", "filtered"], default="full",
                    help="Write full labels (all classes, default) or filtered to the target class only")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-file info")
    args = ap.parse_args()

    dataset = Path(args.dataset_path).resolve()
    class_ids: Set[int] = set(args.class_ids)
    out_root = Path(args.output_dir).resolve() if args.output_dir else dataset.with_name(dataset.name + "_by_class")

    # Validate dataset
    splits = validate_dataset_structure(dataset)
    print(f"Found splits: {', '.join(splits)}")
    print(f"Output root: {out_root}")
    print(f"Classes to extract: {sorted(class_ids)}")
    print(f"Label mode: {args.label_mode}\n")

    # Prepare dirs
    for cid in class_ids:
        ensure_dirs(out_root / f"class_{cid}/images")
        ensure_dirs(out_root / f"class_{cid}/labels")

    # Collect all pairs first (for progress bar total)
    all_pairs = []
    for split in splits:
        all_pairs.extend(list(iter_image_label_pairs(dataset / split)))

    print(f"Scanning {len(all_pairs)} image/label pairs...\n")

    per_class_counts = {cid: 0 for cid in class_ids}

    # Progress bar
    with tqdm(total=len(all_pairs), desc="Processing dataset", unit="file", ncols=100) as pbar:
        for img_path, lbl_path in all_pairs:
            parsed, raw_lines = parse_label_classes(lbl_path)
            if not parsed and not raw_lines:
                pbar.update(1)
                continue

            # Does the label contain any of the requested classes?
            present = {cid for cid, _ in parsed if cid in class_ids}
            if not present:
                pbar.update(1)
                continue

            for target_cid in sorted(present):
                dst_img = out_root / f"class_{target_cid}/images" / img_path.name
                dst_lbl = out_root / f"class_{target_cid}/labels" / lbl_path.name

                try:
                    # Always copy/link the image
                    copy_one(img_path, dst_img, args.copy_mode)

                    # Label writing strategy
                    if args.label_mode == "full":
                        # Preserve original label lines (all classes)
                        write_label(dst_lbl, raw_lines)
                    else:
                        # Filtered: only this class's lines
                        filtered = [line for c, line in parsed if c == target_cid]
                        write_label(dst_lbl, filtered)

                    per_class_counts[target_cid] += 1
                    if args.verbose:
                        tqdm.write(f"[class {target_cid}] {img_path.name}")
                except Exception as e:
                    tqdm.write(f"ERROR writing {img_path.name} for class {target_cid}: {e}")

            pbar.update(1)

    # Summary
    print("\n" + "=" * 60)
    print(f"Processed {len(all_pairs)} image/label pairs.")
    for cid in sorted(class_ids):
        print(f"Class {cid}: {per_class_counts[cid]} images exported")
    print(f"Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
