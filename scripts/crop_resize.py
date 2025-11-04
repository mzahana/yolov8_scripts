#!/usr/bin/env python3
"""
Image Crop and Resize Script (progress-bar edition)

Usage:
    python crop_resize.py /path/to/images [options]

Options:
    --crop-x CROP_TOP_LEFT_X     Top-left X coordinate for cropping (default: None)
    --crop-y CROP_TOP_LEFT_Y     Top-left Y coordinate for cropping (default: None)
    --crop-width CROP_WIDTH      Width of the crop area (default: None)
    --crop-height CROP_HEIGHT    Height of the crop area (default: None)
    --resize-width WIDTH         Width for resizing (default: 640)
    --resize-height HEIGHT       Height for resizing (default: 640)
    --output-dir OUTPUT_DIR      Output directory (default: input_dir + '_processed')

Notes:
- Only WARNING and ERROR logs are shown.
- A progress bar displays overall progress.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import logging
from typing import Optional, Tuple

# ---------- Logging: show only WARN/ERROR by default ----------
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Optional progress bar via tqdm ----------
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        # Minimal fallback if tqdm isn't available: no flooding, no per-item print.
        return iterable


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Crop and resize images in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crop_resize.py /path/to/images
  python crop_resize.py /path/to/images --crop-x 210 --crop-y 48 --crop-width 410 --crop-height 513
  python crop_resize.py /path/to/images --resize-width 800 --resize-height 600
        """
    )

    parser.add_argument('input_dir', help='Path to directory containing images')

    # Crop arguments
    parser.add_argument('--crop-x', type=int, help='Top-left X coordinate for cropping')
    parser.add_argument('--crop-y', type=int, help='Top-left Y coordinate for cropping')
    parser.add_argument('--crop-width', type=int, help='Width of the crop area')
    parser.add_argument('--crop-height', type=int, help='Height of the crop area')

    # Resize arguments
    parser.add_argument('--resize-width', type=int, default=640, help='Width for resizing (default: 640)')
    parser.add_argument('--resize-height', type=int, default=640, help='Height for resizing (default: 640)')

    # Output directory
    parser.add_argument('--output-dir', help='Output directory (default: input_dir + "_processed")')

    return parser.parse_args()


def validate_crop_args(crop_x, crop_y, crop_width, crop_height) -> bool:
    """Validate that all crop arguments are provided together or none at all."""
    crop_args = [crop_x, crop_y, crop_width, crop_height]
    crop_provided = [arg is not None for arg in crop_args]

    if any(crop_provided) and not all(crop_provided):
        logger.error("If cropping, all crop parameters (--crop-x, --crop-y, --crop-width, --crop-height) must be provided")
        sys.exit(1)

    return all(crop_provided)


def get_supported_image_files(directory: Path):
    """Get all supported image files from the directory."""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]


def process_image(
    input_path: Path,
    output_path: Path,
    crop_params: Optional[Tuple[int, int, int, int]] = None,
    resize_dims: Tuple[int, int] = (640, 640),
) -> bool:
    """
    Process a single image: crop (if specified) and resize.

    Returns:
        True if processed and saved successfully; False otherwise.
    """
    try:
        with Image.open(input_path) as img:
            # Crop if parameters are provided
            if crop_params:
                x, y, width, height = crop_params

                # Validate crop dimensions
                if x < 0 or y < 0:
                    logger.warning(f"{input_path.name}: negative crop coordinates; skipping crop")
                elif x + width > img.width or y + height > img.height:
                    logger.warning(f"{input_path.name}: crop area exceeds image bounds; skipping crop")
                else:
                    img = img.crop((x, y, x + width, y + height))

            # Resize the image (LANCZOS)
            img_resized = img.resize(resize_dims, Image.Resampling.LANCZOS)

            # Ensure output directory exists (in case of nested structure later)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the processed image
            img_resized.save(output_path, quality=95, optimize=True)
            return True

    except Exception as e:
        logger.error(f"{input_path.name}: error processing - {e}")
        return False


def main():
    args = parse_arguments()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Validate crop arguments
    should_crop = validate_crop_args(args.crop_x, args.crop_y, args.crop_width, args.crop_height)

    # Set up crop parameters
    crop_params = None
    if should_crop:
        crop_params = (args.crop_x, args.crop_y, args.crop_width, args.crop_height)

    # Set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_processed"
    output_dir.mkdir(exist_ok=True)

    # Get all image files
    image_files = get_supported_image_files(input_dir)
    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        sys.exit(0)

    resize_dims = (args.resize_width, args.resize_height)

    # Progress bar setup
    bar_kwargs = dict(desc="Processing images", unit="img", ncols=80, leave=False)
    if _HAS_TQDM:
        iterator = tqdm(image_files, **bar_kwargs)
    else:
        # Fallback: no progress prints to avoid flooding
        iterator = image_files

    processed_ok = 0
    processed_fail = 0

    for image_file in iterator:
        output_path = output_dir / image_file.name
        ok = process_image(image_file, output_path, crop_params, resize_dims)
        if ok:
            processed_ok += 1
        else:
            processed_fail += 1

    # Make sure the final line is visible when tqdm is used
    if _HAS_TQDM:
        # Force a clean newline after the bar
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Compact summary (single line)
    print(f"Done. Success: {processed_ok} | Failed: {processed_fail} | Output: {output_dir}")

if __name__ == "__main__":
    main()
