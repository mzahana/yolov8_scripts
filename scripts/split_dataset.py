"""
Dataset Splitter for Images and Labels

This script is designed to take a dataset consisting of images and their corresponding label files,
stored in 'images' and 'labels' subdirectories within a specified parent directory, and split it into
training and validation sets according to a user-defined percentage. The resulting datasets are
organized into 'train' and 'valid' subdirectories, each containing 'images' and 'labels' folders for
the split datasets.

Key Features:
- Automatically identifies and processes all .png and .jpg image files in the 'images' subdirectory.
- Preserves the association between each image and its corresponding label file, ensuring that splits
  maintain matching images and labels.
- Allows for a customizable split percentage to determine the size of the training and validation sets.
- Creates a structured output in the parent directory with distinct subdirectories for training and
  validation datasets, facilitating easy use in machine learning projects.

Usage:
To use this script, specify the path to the parent directory containing the 'images' and 'labels'
subdirectories. Also, define the desired percentage of the dataset to be used for training. The script
will automatically handle the random splitting, copying of image and label files into the appropriate
directories.

Example:
parent_dir = '/path/to/parent/directory'
train_percentage = 0.8  # 80% for training, 20% for validation
split_dataset(parent_dir, train_percentage)

Dependencies:
- Python's standard library modules: pathlib, shutil, and random.

Ensure that your dataset is properly organized with all images stored in an 'images' subdirectory and
all corresponding label files (with matching filenames) in a 'labels' subdirectory within the specified
parent directory before running this script.
"""

import os
import shutil
from pathlib import Path
import random

def split_dataset(parent_dir, train_percentage=0.8):
    parent_dir = Path(parent_dir)
    images_dir = parent_dir / "images"
    labels_dir = parent_dir / "labels"

    # Create train and valid directories with images and labels subdirectories
    train_images_dir = parent_dir / "train/images"
    train_labels_dir = parent_dir / "train/labels"
    valid_images_dir = parent_dir / "valid/images"
    valid_labels_dir = parent_dir / "valid/labels"

    for dir in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # List all image files and shuffle
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    random.shuffle(image_files)

    # Calculate split index
    num_images = len(image_files)
    split_index = int(num_images * train_percentage)

    # Split image files into train and valid sets
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]

    # Function to copy files to their new location
    def copy_files(files, target_images_dir, target_labels_dir):
        for img_file in files:
            # Copy image
            shutil.copy(img_file, target_images_dir)
            
            # Determine and copy corresponding label file
            label_file = labels_dir / img_file.with_suffix('.txt').name
            if label_file.exists():
                shutil.copy(label_file, target_labels_dir)

    # Copy files to their respective directories
    copy_files(train_files, train_images_dir, train_labels_dir)
    copy_files(valid_files, valid_images_dir, valid_labels_dir)

    print(f"Dataset split completed: {len(train_files)} images for training, {len(valid_files)} images for validation.")

# Example usage
parent_dir = '/home/mzahana/datasets/Silki/Bundle_Detection.v3i-filtered.yolov8/train'
train_percentage = 0.9  # 80% of the data for training
split_dataset(parent_dir, train_percentage)
