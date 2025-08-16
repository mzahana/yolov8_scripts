"""
Dataset Splitter for Images and Labels - Command Line Interface

This script is designed to take a dataset consisting of images and their corresponding label files,
stored in 'images' and 'labels' subdirectories within a specified parent directory, and split it into
training and validation sets according to a user-defined percentage. The resulting datasets are
organized into 'train' and 'valid' subdirectories, each containing 'images' and 'labels' folders for
the split datasets.

Key Features:
- Automatically identifies and processes all .png and .jpg image files in the 'images' subdirectory.
- Preserves the association between each image and its corresponding label file, ensuring that splits
  maintain matching images and labels.
- Allows for a customizable split percentage via command line arguments.
- Creates a structured output in the parent directory with distinct subdirectories for training and
  validation datasets, facilitating easy use in machine learning projects.
- Command line interface for easy automation and scripting.

Usage:
Run the script from command line with the following arguments:
- parent_dir: Path to the parent directory containing 'images' and 'labels' subdirectories
- --train-percentage (optional): Percentage of data for training (default: 0.8)
- --seed (optional): Random seed for reproducible splits (default: None)

Examples:
python split_dataset.py /path/to/parent/directory
python split_dataset.py /path/to/parent/directory --train-percentage 0.9
python split_dataset.py /path/to/parent/directory --train-percentage 0.85 --seed 42

Dependencies:
- Python's standard library modules: pathlib, shutil, random, and argparse.

Ensure that your dataset is properly organized with all images stored in an 'images' subdirectory and
all corresponding label files (with matching filenames) in a 'labels' subdirectory within the specified
parent directory before running this script.
"""

import os
import shutil
from pathlib import Path
import random
import argparse

def split_dataset(parent_dir, train_percentage=0.8, seed=None):
    """
    Split dataset into training and validation sets.
    
    Args:
        parent_dir (str or Path): Path to parent directory containing 'images' and 'labels'
        train_percentage (float): Percentage of data for training (0.0 to 1.0)
        seed (int, optional): Random seed for reproducible splits
    """
    if seed is not None:
        random.seed(seed)
    
    parent_dir = Path(parent_dir)
    images_dir = parent_dir / "images"
    labels_dir = parent_dir / "labels"

    # Validate input directories
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Create train and valid directories with images and labels subdirectories
    train_images_dir = parent_dir / "train/images"
    train_labels_dir = parent_dir / "train/labels"
    valid_images_dir = parent_dir / "valid/images"
    valid_labels_dir = parent_dir / "valid/labels"

    for dir in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # List all image files and shuffle
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    random.shuffle(image_files)

    # Calculate split index
    num_images = len(image_files)
    split_index = int(num_images * train_percentage)

    # Split image files into train and valid sets
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]

    # Function to copy files to their new location
    def copy_files(files, target_images_dir, target_labels_dir):
        copied_labels = 0
        for img_file in files:
            # Copy image
            shutil.copy(img_file, target_images_dir)
            
            # Determine and copy corresponding label file
            label_file = labels_dir / img_file.with_suffix('.txt').name
            if label_file.exists():
                shutil.copy(label_file, target_labels_dir)
                copied_labels += 1
        return copied_labels

    # Copy files to their respective directories
    train_labels_copied = copy_files(train_files, train_images_dir, train_labels_dir)
    valid_labels_copied = copy_files(valid_files, valid_images_dir, valid_labels_dir)

    print(f"Dataset split completed!")
    print(f"Training set: {len(train_files)} images, {train_labels_copied} labels")
    print(f"Validation set: {len(valid_files)} images, {valid_labels_copied} labels")
    print(f"Split ratio: {train_percentage:.1%} train, {1-train_percentage:.1%} validation")
    if seed is not None:
        print(f"Random seed used: {seed}")

def main():
    parser = argparse.ArgumentParser(
        description="Split image dataset into training and validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset
  %(prog)s /path/to/dataset --train-percentage 0.9
  %(prog)s /path/to/dataset --train-percentage 0.85 --seed 42
        """
    )
    
    parser.add_argument(
        'parent_dir',
        type=str,
        help='Path to parent directory containing "images" and "labels" subdirectories'
    )
    
    parser.add_argument(
        '--train-percentage',
        type=float,
        default=0.8,
        metavar='FLOAT',
        help='Percentage of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        metavar='INT',
        help='Random seed for reproducible splits'
    )
    
    args = parser.parse_args()
    
    # Validate train percentage
    if not 0.0 < args.train_percentage < 1.0:
        parser.error("train-percentage must be between 0.0 and 1.0")
    
    try:
        split_dataset(args.parent_dir, args.train_percentage, args.seed)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())