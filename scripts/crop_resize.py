#!/usr/bin/env python3
"""
Image Crop and Resize Script

This script crops and resizes images in a directory based on command-line arguments.

Usage:
    python image_processor.py /path/to/images [options]

Options:
    --crop-x CROP_TOP_LEFT_X     Top-left X coordinate for cropping (default: None)
    --crop-y CROP_TOP_LEFT_Y     Top-left Y coordinate for cropping (default: None)
    --crop-width CROP_WIDTH      Width of the crop area (default: None)
    --crop-height CROP_HEIGHT    Height of the crop area (default: None)
    --resize-width WIDTH         Width for resizing (default: 640)
    --resize-height HEIGHT       Height for resizing (default: 640)
    --output-dir OUTPUT_DIR      Output directory (default: input_dir + '_processed')

Example:
    python image_processor.py /path/to/images --crop-x 210 --crop-y 48 --crop-width 410 --crop-height 513 --resize-width 800 --resize-height 600
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Crop and resize images in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_processor.py /path/to/images
  python image_processor.py /path/to/images --crop-x 210 --crop-y 48 --crop-width 410 --crop-height 513
  python image_processor.py /path/to/images --resize-width 800 --resize-height 600
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

def validate_crop_args(crop_x, crop_y, crop_width, crop_height):
    """Validate that all crop arguments are provided together or none at all."""
    crop_args = [crop_x, crop_y, crop_width, crop_height]
    crop_provided = [arg is not None for arg in crop_args]
    
    if any(crop_provided) and not all(crop_provided):
        logger.error("If cropping, all crop parameters (--crop-x, --crop-y, --crop-width, --crop-height) must be provided")
        sys.exit(1)
    
    return all(crop_provided)

def get_supported_image_files(directory):
    """Get all supported image files from the directory."""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(file_path)
    
    return image_files

def process_image(input_path, output_path, crop_params=None, resize_dims=(640, 640)):
    """
    Process a single image: crop (if specified) and resize.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        crop_params: Tuple of (x, y, width, height) for cropping, or None
        resize_dims: Tuple of (width, height) for resizing
    """
    try:
        with Image.open(input_path) as img:
            logger.info(f"Processing {input_path.name} - Original size: {img.size}")
            
            # Crop if parameters are provided
            if crop_params:
                x, y, width, height = crop_params
                
                # Validate crop dimensions
                if x < 0 or y < 0:
                    logger.warning(f"Negative crop coordinates for {input_path.name}, skipping crop")
                elif x + width > img.width or y + height > img.height:
                    logger.warning(f"Crop area exceeds image bounds for {input_path.name}, skipping crop")
                else:
                    # Crop the image
                    img = img.crop((x, y, x + width, y + height))
                    logger.info(f"Cropped to: {img.size}")
            
            # Resize the image
            img_resized = img.resize(resize_dims, Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {img_resized.size}")
            
            # Save the processed image
            img_resized.save(output_path, quality=95, optimize=True)
            logger.info(f"Saved: {output_path}")
            
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {str(e)}")

def main():
    """Main function."""
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
        logger.info(f"Crop parameters: x={args.crop_x}, y={args.crop_y}, width={args.crop_width}, height={args.crop_height}")
    else:
        logger.info("No cropping will be applied")
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_processed"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get all image files
    image_files = get_supported_image_files(input_dir)
    
    if not image_files:
        logger.warning(f"No supported image files found in {input_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process each image
    resize_dims = (args.resize_width, args.resize_height)
    logger.info(f"Resize dimensions: {resize_dims}")
    
    processed_count = 0
    for image_file in image_files:
        output_path = output_dir / image_file.name
        process_image(image_file, output_path, crop_params, resize_dims)
        processed_count += 1
    
    logger.info(f"Processing complete! {processed_count} images processed and saved to {output_dir}")

if __name__ == "__main__":
    main()
