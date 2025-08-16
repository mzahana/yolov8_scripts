#!/usr/bin/env python3
"""
YOLO Dataset Random Sampler

This script randomly samples images from a YOLO format dataset while maintaining 
the original folder structure. It supports sampling by number of images or percentage
and preserves corresponding label files and data.yaml configuration.

Features:
- Random sampling by count or percentage
- Maintains train/valid/test split structure
- Copies corresponding label files
- Preserves data.yaml configuration
- Command line interface with argument validation

Requirements:
- Dataset must have at least 'train' and 'valid' folders
- Each split folder must contain 'images' and 'labels' subfolders
- data.yaml file must be present in the dataset root

Author: Assistant
Version: 1.0

Usage Examples:
    # Sample 1000 random images from each split
    python sample_yolo_dataset.py /path/to/dataset --count 1000
    
    # Sample 15% of images from each split
    python sample_yolo_dataset.py /path/to/dataset --percentage 15
    
    # Sample with custom output directory
    python sample_yolo_dataset.py /path/to/dataset --count 500 --output_dir /path/to/output
    
    # Sample 20% with verbose output
    python sample_yolo_dataset.py /path/to/dataset --percentage 20 --verbose
"""

import os
import argparse
import random
import shutil
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

def validate_dataset_structure(dataset_path: Path) -> List[str]:
    """
    Validate that the dataset has the required YOLO structure.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        List of available splits
        
    Raises:
        ValueError: If required structure is not found
    """
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Check for data.yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise ValueError(f"data.yaml not found in {dataset_path}")
    
    # Check for required splits
    required_splits = ['train', 'valid']
    optional_splits = ['test']
    available_splits = []
    
    for split in required_splits + optional_splits:
        split_path = dataset_path / split
        if split_path.exists():
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            if not images_path.exists():
                raise ValueError(f"Missing images folder in {split} split")
            if not labels_path.exists():
                raise ValueError(f"Missing labels folder in {split} split")
            
            available_splits.append(split)
    
    # Check that at least train and valid exist
    if 'train' not in available_splits or 'valid' not in available_splits:
        raise ValueError("Dataset must have at least 'train' and 'valid' splits")
    
    return available_splits

def get_image_label_pairs(split_path: Path) -> List[Tuple[Path, Path]]:
    """
    Get corresponding image and label file pairs from a split directory.
    
    Args:
        split_path: Path to the split directory (e.g., train, valid, test)
        
    Returns:
        List of (image_path, label_path) tuples
    """
    images_path = split_path / "images"
    labels_path = split_path / "labels"
    
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for img_file in images_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            # Find corresponding label file
            label_file = labels_path / (img_file.stem + '.txt')
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"Warning: No label found for {img_file.name}")
    
    return pairs

def sample_pairs(pairs: List[Tuple[Path, Path]], count: Optional[int] = None, 
                percentage: Optional[float] = None) -> List[Tuple[Path, Path]]:
    """
    Sample pairs based on count or percentage.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        count: Number of pairs to sample
        percentage: Percentage of pairs to sample
        
    Returns:
        Sampled list of pairs
    """
    if count is not None:
        sample_size = min(count, len(pairs))
    elif percentage is not None:
        sample_size = int(len(pairs) * percentage / 100)
    else:
        raise ValueError("Either count or percentage must be specified")
    
    return random.sample(pairs, sample_size)

def copy_sampled_data(sampled_pairs: List[Tuple[Path, Path]], output_split_path: Path, 
                     verbose: bool = False) -> None:
    """
    Copy sampled image and label pairs to output directory.
    
    Args:
        sampled_pairs: List of (image_path, label_path) tuples to copy
        output_split_path: Output split directory path
        verbose: Whether to print copy operations
    """
    output_images = output_split_path / "images"
    output_labels = output_split_path / "labels"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    for img_path, label_path in sampled_pairs:
        # Copy image
        shutil.copy2(img_path, output_images / img_path.name)
        # Copy label
        shutil.copy2(label_path, output_labels / label_path.name)
        
        if verbose:
            print(f"Copied: {img_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample images from a YOLO format dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset --count 1000
  %(prog)s /path/to/dataset --percentage 15
  %(prog)s /path/to/dataset --count 500 --output_dir /path/to/output --verbose
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to the YOLO dataset directory'
    )
    
    # Sampling options (mutually exclusive)
    sampling_group = parser.add_mutually_exclusive_group(required=True)
    sampling_group.add_argument(
        '--count', '-c',
        type=int,
        help='Number of random images to sample from each split'
    )
    sampling_group.add_argument(
        '--percentage', '-p',
        type=float,
        help='Percentage of images to sample from each split (0-100)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        help='Output directory for sampled dataset (default: parent_dir/original_name_sampled)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.percentage is not None and (args.percentage <= 0 or args.percentage > 100):
        parser.error("Percentage must be between 0 and 100")
    
    if args.count is not None and args.count <= 0:
        parser.error("Count must be positive")
    
    # Set random seed
    random.seed(args.seed)
    
    # Convert paths
    dataset_path = Path(args.dataset_path)
    
    try:
        # Validate dataset structure
        print(f"Validating dataset structure at {dataset_path}")
        available_splits = validate_dataset_structure(dataset_path)
        print(f"Found splits: {', '.join(available_splits)}")
        
        # Determine output directory
        if args.output_dir:
            output_path = Path(args.output_dir)
        else:
            parent_dir = dataset_path.parent
            dataset_name = dataset_path.name
            if args.count:
                suffix = f"sampled_{args.count}"
            else:
                suffix = f"sampled_{args.percentage}pct"
            output_path = parent_dir / f"{dataset_name}_{suffix}"
        
        print(f"Output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy data.yaml
        shutil.copy2(dataset_path / "data.yaml", output_path / "data.yaml")
        print("Copied data.yaml")
        
        # Process each split
        total_original = 0
        total_sampled = 0
        
        for split in available_splits:
            print(f"\nProcessing {split} split...")
            
            # Get image-label pairs
            pairs = get_image_label_pairs(dataset_path / split)
            total_original += len(pairs)
            
            if not pairs:
                print(f"Warning: No valid image-label pairs found in {split} split")
                continue
            
            # Sample pairs
            sampled_pairs = sample_pairs(pairs, args.count, args.percentage)
            total_sampled += len(sampled_pairs)
            
            print(f"Sampling {len(sampled_pairs)} from {len(pairs)} pairs in {split}")
            
            # Copy sampled data
            output_split_path = output_path / split
            copy_sampled_data(sampled_pairs, output_split_path, args.verbose)
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Sampling completed successfully!")
        print(f"Total original images: {total_original}")
        print(f"Total sampled images: {total_sampled}")
        if total_original > 0:
            print(f"Sampling ratio: {total_sampled/total_original*100:.1f}%")
        print(f"Output saved to: {output_path}")
        print(f"{'='*50}")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())