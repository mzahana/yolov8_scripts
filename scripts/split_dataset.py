"""
Dataset Splitter for Images and Labels - YOLO Compatible Command Line Interface

This script splits a dataset consisting of images and their corresponding label files into
training, validation, and test sets according to user-defined percentages. The resulting 
datasets are organized in a YOLO-compatible structure with 'train', 'valid', and 'test' 
subdirectories, each containing 'images' and 'labels' folders.

Key Features:
- Automatically identifies and processes all common image formats (.png, .jpg, .jpeg, .bmp, .tiff)
- Preserves the association between each image and its corresponding label file
- Creates YOLO-compatible dataset structure with train/valid/test splits
- Customizable split percentages via command line arguments
- Optional YAML file generation for YOLO training configuration
- Detailed statistics and optional file listing output
- Support for reproducible splits with random seed
- Validation of label files and reporting of missing labels

Usage:
Run the script from command line with the following arguments:
- parent_dir: Path to the parent directory containing 'images' and 'labels' subdirectories
- --train (optional): Percentage for training (default: 0.7)
- --valid (optional): Percentage for validation (default: 0.2)
- --test (optional): Percentage for testing (default: 0.1)
- --seed (optional): Random seed for reproducible splits
- --yaml (optional): Generate YOLO config YAML file
- --verbose (optional): Show detailed file listings

Examples:
python split_dataset.py /path/to/dataset
python split_dataset.py /path/to/dataset --train 0.8 --valid 0.15 --test 0.05
python split_dataset.py /path/to/dataset --train 0.7 --valid 0.2 --test 0.1 --seed 42
python split_dataset.py /path/to/dataset --yaml --verbose

Dependencies:
- Python's standard library modules: pathlib, shutil, random, argparse, yaml (optional)
"""

import os
import shutil
from pathlib import Path
import random
import argparse
import json
from typing import List, Tuple, Dict
import sys


class YOLODatasetSplitter:
    """Class to handle YOLO dataset splitting operations."""
    
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def __init__(self, parent_dir: Path, train_pct: float = 0.7, 
                 valid_pct: float = 0.2, test_pct: float = 0.1, 
                 seed: int = None, verbose: bool = False):
        """
        Initialize the dataset splitter.
        
        Args:
            parent_dir: Path to parent directory containing 'images' and 'labels'
            train_pct: Percentage for training set
            valid_pct: Percentage for validation set  
            test_pct: Percentage for test set
            seed: Random seed for reproducible splits
            verbose: Whether to print detailed information
        """
        self.parent_dir = Path(parent_dir)
        self.train_pct = train_pct
        self.valid_pct = valid_pct
        self.test_pct = test_pct
        self.seed = seed
        self.verbose = verbose
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Validate percentages
        total = train_pct + valid_pct + test_pct
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split percentages must sum to 1.0, got {total:.3f}")
        
        # Setup directories
        self.images_dir = self.parent_dir / "images"
        self.labels_dir = self.parent_dir / "labels"
        
        # Statistics tracking
        self.stats = {
            'train': {'images': 0, 'labels': 0, 'missing_labels': []},
            'valid': {'images': 0, 'labels': 0, 'missing_labels': []},
            'test': {'images': 0, 'labels': 0, 'missing_labels': []}
        }
    
    def validate_directories(self):
        """Validate that required input directories exist."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
    
    def create_output_directories(self) -> Dict[str, Dict[str, Path]]:
        """Create output directory structure for train/valid/test splits."""
        dirs = {}
        for split in ['train', 'valid', 'test']:
            dirs[split] = {
                'images': self.parent_dir / split / 'images',
                'labels': self.parent_dir / split / 'labels'
            }
            # Create directories
            dirs[split]['images'].mkdir(parents=True, exist_ok=True)
            dirs[split]['labels'].mkdir(parents=True, exist_ok=True)
        return dirs
    
    def get_image_files(self) -> List[Path]:
        """Get all image files from the images directory."""
        image_files = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
            image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.images_dir}")
        
        return sorted(image_files)  # Sort for consistency
    
    def split_files(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split files into train, valid, and test sets."""
        # Shuffle files
        files_copy = files.copy()
        random.shuffle(files_copy)
        
        # Calculate split indices
        n = len(files_copy)
        train_idx = int(n * self.train_pct)
        valid_idx = train_idx + int(n * self.valid_pct)
        
        # Split files
        train_files = files_copy[:train_idx]
        valid_files = files_copy[train_idx:valid_idx]
        test_files = files_copy[valid_idx:]
        
        return train_files, valid_files, test_files
    
    def copy_files(self, files: List[Path], split_name: str, 
                   target_dirs: Dict[str, Path]) -> int:
        """
        Copy image and label files to target directories.
        
        Returns:
            Number of successfully copied label files
        """
        labels_copied = 0
        missing_labels = []
        
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, target_dirs['images'])
            self.stats[split_name]['images'] += 1
            
            # Find and copy corresponding label file
            label_name = img_file.stem + '.txt'
            label_file = self.labels_dir / label_name
            
            if label_file.exists():
                shutil.copy2(label_file, target_dirs['labels'])
                labels_copied += 1
                self.stats[split_name]['labels'] += 1
            else:
                missing_labels.append(img_file.name)
                self.stats[split_name]['missing_labels'].append(img_file.name)
        
        if missing_labels and self.verbose:
            print(f"  Warning: {len(missing_labels)} missing label files in {split_name} set")
            
        return labels_copied
    
    def generate_yaml_config(self, num_classes: int = None):
        """Generate YOLO configuration YAML file."""
        yaml_content = {
            'path': str(self.parent_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
        }
        
        # Try to detect number of classes from a sample label file
        if num_classes is None:
            num_classes = self._detect_num_classes()
        
        if num_classes:
            yaml_content['nc'] = num_classes
            yaml_content['names'] = [f'class_{i}' for i in range(num_classes)]
        
        # Write YAML file
        yaml_path = self.parent_dir / 'dataset.yaml'
        try:
            import yaml
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            print(f"\nGenerated YOLO config file: {yaml_path}")
        except ImportError:
            # Fallback to JSON-like format if PyYAML not available
            with open(yaml_path, 'w') as f:
                f.write(f"# YOLO Dataset Configuration\n")
                f.write(f"path: {yaml_content['path']}\n")
                f.write(f"train: {yaml_content['train']}\n")
                f.write(f"val: {yaml_content['val']}\n")
                f.write(f"test: {yaml_content['test']}\n")
                if num_classes:
                    f.write(f"nc: {num_classes}\n")
                    f.write(f"names: {yaml_content['names']}\n")
            print(f"\nGenerated YOLO config file (simple format): {yaml_path}")
    
    def _detect_num_classes(self) -> int:
        """Detect number of classes from label files."""
        max_class = -1
        label_files = list(self.labels_dir.glob('*.txt'))[:100]  # Sample first 100
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            max_class = max(max_class, class_id)
            except (ValueError, IndexError):
                continue
        
        return max_class + 1 if max_class >= 0 else None
    
    def print_statistics(self):
        """Print detailed statistics about the split."""
        print("\n" + "="*60)
        print("DATASET SPLIT STATISTICS")
        print("="*60)
        
        total_images = sum(s['images'] for s in self.stats.values())
        total_labels = sum(s['labels'] for s in self.stats.values())
        
        for split_name in ['train', 'valid', 'test']:
            split_stats = self.stats[split_name]
            pct = getattr(self, f'{split_name}_pct' if split_name != 'valid' else 'valid_pct')
            
            print(f"\n{split_name.upper()} SET ({pct:.1%}):")
            print(f"  Images: {split_stats['images']:,} files")
            print(f"  Labels: {split_stats['labels']:,} files")
            
            if split_stats['missing_labels']:
                print(f"  Missing labels: {len(split_stats['missing_labels'])} files")
                if self.verbose and len(split_stats['missing_labels']) <= 10:
                    for fname in split_stats['missing_labels'][:10]:
                        print(f"    - {fname}")
        
        print(f"\nTOTAL:")
        print(f"  Images: {total_images:,} files")
        print(f"  Labels: {total_labels:,} files")
        print(f"  Label coverage: {total_labels/total_images:.1%}")
        
        if self.seed is not None:
            print(f"\nRandom seed: {self.seed}")
        
        print("="*60)
    
    def run(self, generate_yaml: bool = False):
        """Execute the dataset splitting process."""
        print(f"Starting dataset split in: {self.parent_dir}")
        
        # Validate input
        self.validate_directories()
        
        # Get image files
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} image files")
        
        # Create output directories
        output_dirs = self.create_output_directories()
        
        # Split files
        train_files, valid_files, test_files = self.split_files(image_files)
        
        # Copy files to respective directories
        print("\nCopying files...")
        self.copy_files(train_files, 'train', output_dirs['train'])
        self.copy_files(valid_files, 'valid', output_dirs['valid'])
        self.copy_files(test_files, 'test', output_dirs['test'])
        
        # Generate YAML if requested
        if generate_yaml:
            self.generate_yaml_config()
        
        # Print statistics
        self.print_statistics()
        
        print("\n✓ Dataset split completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Split image dataset into train/valid/test sets for YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset
  %(prog)s /path/to/dataset --train 0.8 --valid 0.15 --test 0.05
  %(prog)s /path/to/dataset --seed 42 --yaml
  %(prog)s /path/to/dataset --train 0.6 --valid 0.2 --test 0.2 --verbose
        """
    )
    
    parser.add_argument(
        'parent_dir',
        type=str,
        help='Path to parent directory containing "images" and "labels" subdirectories'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        metavar='PCT',
        help='Percentage for training set (default: 0.7)'
    )
    
    parser.add_argument(
        '--valid',
        type=float,
        default=0.2,
        metavar='PCT',
        help='Percentage for validation set (default: 0.2)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Percentage for test set (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        metavar='INT',
        help='Random seed for reproducible splits'
    )
    
    parser.add_argument(
        '--yaml',
        action='store_true',
        help='Generate YOLO configuration YAML file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information during processing'
    )
    
    args = parser.parse_args()
    
    # Validate percentages
    for pct_name, pct_value in [('train', args.train), ('valid', args.valid), ('test', args.test)]:
        if not 0.0 < pct_value <= 1.0:
            parser.error(f"{pct_name} percentage must be between 0.0 and 1.0")
    
    total = args.train + args.valid + args.test
    if abs(total - 1.0) > 0.001:
        parser.error(f"Percentages must sum to 1.0, got {total:.3f}")
    
    try:
        splitter = YOLODatasetSplitter(
            parent_dir=args.parent_dir,
            train_pct=args.train,
            valid_pct=args.valid,
            test_pct=args.test,
            seed=args.seed,
            verbose=args.verbose
        )
        splitter.run(generate_yaml=args.yaml)
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())