#!/usr/bin/env python3
import argparse, os, sys, glob, random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2

from ultralytics import YOLO

# --- Utilities ---------------------------------------------------------------

def list_images(root, recursive, exts):
    if recursive:
        paths = []
        for e in exts:
            paths += glob.glob(str(Path(root) / f"**/*.{e}"), recursive=True)
    else:
        paths = []
        for e in exts:
            paths += glob.glob(str(Path(root) / f"*.{e}"))
    # unique & stable
    paths = sorted(set(paths))
    return paths

def letterbox(im, new_shape=640, color=(114,114,114)):
    """
    Minimal letterbox to match YOLO preprocessing:
    - keep aspect ratio, pad to square new_shape x new_shape
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = im.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded

def collate_batch(paths, imgsz, device):
    """Load a batch of images -> tensor [B,3,H,W] normalized to [0,1] (RGB)."""
    batch = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = letterbox(im, imgsz)
        im = torch.from_numpy(im).to(device)
        im = im.permute(2,0,1).float() / 255.0
        batch.append(im)
    if not batch:
        return None
    return torch.stack(batch, dim=0)

def set_bn_cumulative(m: nn.Module):
    for module in m.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()   # forget old stats
            module.momentum = None         # cumulative moving average

def restore_bn_momentum(m: nn.Module, momentum=0.1):
    for module in m.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("BatchNorm Recalibration (AdaBN) for Ultralytics YOLO")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to trained YOLO .pt (e.g., runs/detect/train/weights/best.pt)")
    ap.add_argument("--images", type=str, required=True,
                    help="Folder with new-site frames (jpg/png)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (letterboxed)")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Max number of images to use (shuffle applied before limiting)")
    ap.add_argument("--exts", type=str, default="jpg,jpeg,png", help="Comma-separated extensions")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="cuda[:id] or cpu")
    ap.add_argument("--save", type=str, default=None,
                help="Output path for recalibrated weights (.pt)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the image list before limiting")
    ap.add_argument("--workers", type=int, default=0, help="(reserved)")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load model
    yolo = YOLO(args.model)
    model = yolo.model.to(device)

    # 2) Freeze parameters (we won't backprop)
    for p in model.parameters():
        p.requires_grad_(False)

    # 3) Put model into train mode to enable BN updates
    model.train()

    # 4) Reset BN stats and use cumulative momentum
    set_bn_cumulative(model)

    # 5) Collect images
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    all_imgs = list_images(args.images, args.recursive, exts)
    if not all_imgs:
        print(f"[ERROR] No images found in: {args.images}")
        sys.exit(1)

    if args.shuffle:
        random.shuffle(all_imgs)
    if args.limit > 0:
        all_imgs = all_imgs[:args.limit]

    print(f"[INFO] Recalibrating BN with {len(all_imgs)} images "
          f"(batch={args.batch}, imgsz={args.imgsz})")

    # 6) Forward passes (no gradients)
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    model.eval()  # eval for layers like dropout; but BN stats won't update in eval
    # Important: BN updates only happen in train() mode. So use train() during passes:
    model.train()

    with torch.no_grad():
        # small warmup for CUDA kernels
        dummy = torch.zeros(1,3,args.imgsz,args.imgsz, device=device, dtype=torch.float32)
        _ = model(dummy)

        # iterate in batches
        for i in range(0, len(all_imgs), args.batch):
            batch_paths = all_imgs[i : i + args.batch]
            batch = collate_batch(batch_paths, args.imgsz, device)
            if batch is None:
                continue
            # optional autocast – safe for BN stats (means/vars computed in float)
            # but to be conservative for BN numerics, avoid autocast on FP16 here:
            batch = batch.to(dtype=torch.float32)
            _ = model(batch)

    # 7) Switch back to eval and restore standard BN momentum
    model.eval()
    restore_bn_momentum(model, momentum=0.1)

    # 8) Determine output path
    if args.save:  # user explicitly provided output path
        save_path = Path(args.save).expanduser().resolve()
    else:
        input_path = Path(args.model).expanduser().resolve()
        save_path = input_path.with_name(f"{input_path.stem}_bn_calibrated{input_path.suffix}")

    # 9) Save in Ultralytics checkpoint format (not just state_dict)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {'model': model.cpu()}
    torch.save(ckpt, save_path)

    # 10) Print summary
    print("\n" + "="*72)
    print("[✅] BN-calibrated model saved successfully!")
    print(f"      → Full path: {save_path}")
    print(f"      → Next step: yolo export model={save_path} format=engine imgsz={args.imgsz} half=1")
    print("="*72 + "\n")



if __name__ == "__main__":
    main()
