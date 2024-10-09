"""
Merge Datasets Script

This script merges multiple datasets into one consolidated dataset while preserving the directory structure.
Each input dataset must have the same structure, with subdirectories for train, val, and test splits,
each containing images and corresponding labels. The merged dataset will also have a unified data.yaml file.

Usage:
    python merge_datasets.py /path/to/dataset1 /path/to/dataset2 /path/to/output_dir

Copyright (c) Mohamed Abdelakder 2024
"""

import os
import shutil
import argparse
import yaml
from collections import defaultdict

def merge_datasets(dataset_dirs, output_dir):
    # Initialize split size info
    split_sizes = defaultdict(int)
    
    print("Creating output directories...")
    # Create output directories
    for split in ["train", "valid", "test"]:
        images_path = os.path.join(output_dir, split, "images")
        labels_path = os.path.join(output_dir, split, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

    print("Merging datasets...")
    # Merge datasets
    for dataset_dir in dataset_dirs:
        print(f"Processing dataset: {dataset_dir}")
        for split in ["train", "valid", "test"]:
            images_src = os.path.join(dataset_dir, split, "images")
            labels_src = os.path.join(dataset_dir, split, "labels")
            images_dst = os.path.join(output_dir, split, "images")
            labels_dst = os.path.join(output_dir, split, "labels")

            if os.path.exists(images_src) and os.path.exists(labels_src):
                print(f"  Merging {split} split from {dataset_dir}...")
                # Copy images
                for img_file in os.listdir(images_src):
                    src_path = os.path.join(images_src, img_file)
                    dst_path = os.path.join(images_dst, img_file)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        split_sizes[split] += 1

                # Copy labels
                for label_file in os.listdir(labels_src):
                    src_path = os.path.join(labels_src, label_file)
                    dst_path = os.path.join(labels_dst, label_file)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)

    # Print split size info
    print("Merging complete. Split sizes:")
    for split, size in split_sizes.items():
        print(f"  {split} split size: {size} images")

    # Merge data.yaml files
    print("Merging data.yaml file...")
    yaml_path = os.path.join(dataset_dirs[0], "data.yaml")
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)

    output_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(output_yaml_path, 'w') as file:
        yaml.dump(data_yaml, file)

    print(f"Merged dataset created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple datasets into one.")
    parser.add_argument("datasets", nargs='+', help="Paths to the dataset directories to be merged.")
    parser.add_argument("output", help="Path to the output directory for the merged dataset.")
    args = parser.parse_args()

    merge_datasets(args.datasets, args.output)


if __name__ == "__main__":
    main()