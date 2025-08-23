#!/usr/bin/env python3
"""
YOLO Dataset Visualizer
Visualizes random images from YOLO dataset with polygon annotations overlaid.

Usage: python yolo_visualizer.py <dataset_path> [--num_images N] [--figsize W H]
"""

import os
import yaml
import random
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import colorsys

def load_data_yaml(dataset_path):
    """Load and parse the data.yaml file."""
    yaml_path = Path(dataset_path) / 'data.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data

def get_split_paths(dataset_path, data_config):
    """Get the paths for train, test, and validation splits."""
    splits = {}
    base_path = Path(dataset_path)
    
    # Check for each split
    for split_name in ['train', 'test', 'val']:
        # Try different possible directory names
        possible_names = [split_name, 'valid' if split_name == 'val' else split_name]
        
        for name in possible_names:
            split_dir = base_path / name
            if split_dir.exists():
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if images_dir.exists() and labels_dir.exists():
                    splits[split_name] = {
                        'images': images_dir,
                        'labels': labels_dir
                    }
                    break
    
    return splits

def get_image_files(images_dir):
    """Get all image files from the images directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    return image_files

def load_annotations(label_path):
    """Load annotations from a label file."""
    annotations = []
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coordinates)
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Group coordinates into (x, y) pairs
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            
            annotations.append({
                'class_id': class_id,
                'points': points
            })
    
    return annotations

def denormalize_coordinates(points, img_width, img_height):
    """Convert normalized coordinates to pixel coordinates."""
    return [(x * img_width, y * img_height) for x, y in points]

def generate_colors(num_classes):
    """Generate distinct colors for each class."""
    colors = []
    for i in range(num_classes):
        hue = i / max(num_classes, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)
    return colors

def visualize_image(image_path, annotations, class_names, colors):
    """Visualize a single image with its annotations."""
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(f"{image_path.name}", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Draw annotations
    for ann in annotations:
        class_id = ann['class_id']
        points = ann['points']
        
        if class_id >= len(class_names):
            continue
        
        # Denormalize coordinates
        pixel_points = denormalize_coordinates(points, img_width, img_height)
        
        # Create polygon
        if len(pixel_points) >= 3:
            polygon = patches.Polygon(
                pixel_points,
                closed=True,
                linewidth=2,
                edgecolor=colors[class_id],
                facecolor=colors[class_id],
                alpha=0.3
            )
            ax.add_patch(polygon)
            
            # Add class label
            center_x = sum(p[0] for p in pixel_points) / len(pixel_points)
            center_y = sum(p[1] for p in pixel_points) / len(pixel_points)
            
            ax.text(
                center_x, center_y,
                class_names[class_id],
                color='white',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[class_id], alpha=0.8)
            )
    
    return fig

def visualize_dataset(dataset_path, num_images=5, figsize=(15, 10)):
    """Main function to visualize the dataset."""
    print(f"Loading dataset from: {dataset_path}")
    
    # Load data configuration
    try:
        data_config = load_data_yaml(dataset_path)
        class_names = data_config.get('names', [])
        num_classes = data_config.get('nc', len(class_names))
        
        print(f"Dataset info:")
        print(f"  Classes ({num_classes}): {class_names}")
        
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        return
    
    # Get split paths
    splits = get_split_paths(dataset_path, data_config)
    if not splits:
        print("No valid splits found!")
        return
    
    print(f"Found splits: {list(splits.keys())}")
    
    # Generate colors for classes
    colors = generate_colors(num_classes)
    
    # Visualize each split
    for split_name, paths in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Get all image files
        image_files = get_image_files(paths['images'])
        print(f"  Found {len(image_files)} images")
        
        if not image_files:
            print(f"  No images found in {split_name} split")
            continue
        
        # Select random images
        n_display = min(num_images, len(image_files))
        selected_images = random.sample(image_files, n_display)
        
        print(f"  Visualizing {n_display} random images")
        
        # Calculate grid layout
        cols = min(3, n_display)
        rows = (n_display + cols - 1) // cols
        
        # Create subplot figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'{split_name.upper()} Split - Random Sample', fontsize=16, fontweight='bold')
        
        # Make axes iterable
        if n_display == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if n_display > 1 else axes
        
        for idx, image_path in enumerate(selected_images):
            # Load annotations
            label_path = paths['labels'] / f"{image_path.stem}.txt"
            annotations = load_annotations(label_path)
            
            # Load and display image
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            ax = axes_flat[idx]
            ax.imshow(image)
            ax.set_title(f"{image_path.name}\n({len(annotations)} objects)", fontsize=10)
            ax.axis('off')
            
            # Draw annotations
            for ann in annotations:
                class_id = ann['class_id']
                points = ann['points']
                
                if class_id >= len(class_names):
                    continue
                
                # Denormalize coordinates
                pixel_points = denormalize_coordinates(points, img_width, img_height)
                
                # Create polygon
                if len(pixel_points) >= 3:
                    polygon = patches.Polygon(
                        pixel_points,
                        closed=True,
                        linewidth=1.5,
                        edgecolor=colors[class_id],
                        facecolor=colors[class_id],
                        alpha=0.3
                    )
                    ax.add_patch(polygon)
                    
                    # Add class label
                    center_x = sum(p[0] for p in pixel_points) / len(pixel_points)
                    center_y = sum(p[1] for p in pixel_points) / len(pixel_points)
                    
                    ax.text(
                        center_x, center_y,
                        class_names[class_id],
                        color='white',
                        fontsize=8,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[class_id], alpha=0.8)
                    )
        
        # Hide empty subplots
        for idx in range(n_display, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\nVisualization complete!")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO dataset with polygon annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yolo_visualizer.py /path/to/dataset
  python yolo_visualizer.py /path/to/dataset --num_images 8
  python yolo_visualizer.py /path/to/dataset --num_images 10 --figsize 20 15
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to the YOLO dataset directory containing data.yaml'
    )
    
    parser.add_argument(
        '--num_images', '-n',
        type=int,
        default=5,
        help='Number of random images to visualize per split (default: 5)'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[15, 10],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 15 10)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        return 1
    
    if not dataset_path.is_dir():
        print(f"Error: '{args.dataset_path}' is not a directory")
        return 1
    
    # Run visualization
    try:
        visualize_dataset(
            dataset_path=args.dataset_path,
            num_images=args.num_images,
            figsize=tuple(args.figsize)
        )
        return 0
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())