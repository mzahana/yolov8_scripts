# Dataset Analytics Script for Segmentation Task
# 
# This script provides analytics for a segmentation dataset, including:
# 1. Counting the number of images in train, validation, and test splits.
# 2. Counting the number of objects per class in each split.
# 3. Generating a heatmap of average object locations in the images.
# 4. Plotting the total number of objects in each dataset split.
# 5. Saving generated plots and analytics in a subdirectory inside the dataset directory.
# 
# Usage:
# python dataset_analytics.py --data_dir <path_to_dataset_directory>
# Example:
# python dataset_analytics.py --data_dir ./dataset
# 
# Copyright Mohamed Abdelkader 2024

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor

# Parse arguments
parser = argparse.ArgumentParser(description="Dataset Analytics for Segmentation Task")
parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
args = parser.parse_args()

# Define paths
data_yaml_path = os.path.join(args.data_dir, "data.yaml")
train_images_path = os.path.join(args.data_dir, "train/images")
val_images_path = os.path.join(args.data_dir, "valid/images")
test_images_path = os.path.join(args.data_dir, "test/images")
train_labels_path = os.path.join(args.data_dir, "train/labels")
val_labels_path = os.path.join(args.data_dir, "valid/labels")
test_labels_path = os.path.join(args.data_dir, "test/labels")

# Load class names from data.yaml
with open(data_yaml_path, 'r') as file:
    data = yaml.safe_load(file)
class_names = data['names']

# Create output directory
output_dir = os.path.join(args.data_dir, "analytics_output")
os.makedirs(output_dir, exist_ok=True)

# Utility function to count images
def count_images(path):
    return len(glob.glob(os.path.join(path, "*.jpg"))) + len(glob.glob(os.path.join(path, "*.png")))

# Utility function to count objects per class
def count_objects_in_file(txt_file):
    class_count = defaultdict(int)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id = int(line.strip().split()[0])
            class_count[class_id] += 1
    return class_count

def count_objects(labels_path):
    class_count = defaultdict(int)
    txt_files = glob.glob(os.path.join(labels_path, "*.txt"))
    with ThreadPoolExecutor() as executor:
        results = executor.map(count_objects_in_file, txt_files)
        for result in results:
            for class_id, count in result.items():
                class_count[class_id] += count
    return class_count

# Image count statistics
train_image_count = count_images(train_images_path)
val_image_count = count_images(val_images_path)
test_image_count = count_images(test_images_path)

total_images = train_image_count + val_image_count + test_image_count

print(f"Total images: {total_images}")
print(f"Train images: {train_image_count}")
print(f"Validation images: {val_image_count}")
print(f"Test images: {test_image_count}")

# Object count statistics
train_object_count = count_objects(train_labels_path)
val_object_count = count_objects(val_labels_path)
test_object_count = count_objects(test_labels_path)

print("\nObjects count per class in Train split:")
for class_id, count in train_object_count.items():
    print(f"Class {class_names[class_id]}: {count}")

print("\nObjects count per class in Validation split:")
for class_id, count in val_object_count.items():
    print(f"Class {class_names[class_id]}: {count}")

print("\nObjects count per class in Test split:")
for class_id, count in test_object_count.items():
    print(f"Class {class_names[class_id]}: {count}")

# Function to update heatmap for a given label file
def update_heatmap(txt_file, heatmap):
    count = 0
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            polygon_points = list(map(float, parts[1:]))
            x_coords = polygon_points[::2]
            y_coords = polygon_points[1::2]
            
            x_mean = int(np.mean(x_coords) * 100)
            y_mean = int(np.mean(y_coords) * 100)
            
            if 0 <= x_mean < 100 and 0 <= y_mean < 100:
                heatmap[y_mean, x_mean] += 1
            count += 1
    return count

# Function to calculate heatmap in parallel
def calculate_heatmap(labels_path):
    heatmap = np.zeros((100, 100))
    txt_files = glob.glob(os.path.join(labels_path, "*.txt"))
    total_count = 0
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda txt_file: update_heatmap(txt_file, heatmap), txt_files)
        for count in results:
            total_count += count
    heatmap /= (total_count + 1e-5)  # Normalizing heatmap
    return heatmap

# Calculate heatmaps
heatmap_train = calculate_heatmap(train_labels_path)
heatmap_val = calculate_heatmap(val_labels_path)

# Plot all statistics in a single figure
fig, axes = plt.subplots(3, 2, figsize=(20, 22))

# Plot object counts per class for Train split
bars = axes[0, 0].bar(class_names, [train_object_count.get(i, 0) for i in range(len(class_names))], color='skyblue')
axes[0, 0].set_xlabel("Class Names")
axes[0, 0].set_ylabel("Number of Objects")
axes[0, 0].set_title("Object Counts per Class in Train Split")
axes[0, 0].tick_params(axis='x', rotation=45)
for bar in bars:
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())), ha='center', va='bottom')

# Plot object counts per class for Validation split
bars = axes[0, 1].bar(class_names, [val_object_count.get(i, 0) for i in range(len(class_names))], color='skyblue')
axes[0, 1].set_xlabel("Class Names")
axes[0, 1].set_ylabel("Number of Objects")
axes[0, 1].set_title("Object Counts per Class in Validation Split")
axes[0, 1].tick_params(axis='x', rotation=45)
for bar in bars:
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())), ha='center', va='bottom')

# Plot object counts per class for Test split
bars = axes[1, 0].bar(class_names, [test_object_count.get(i, 0) for i in range(len(class_names))], color='skyblue')
axes[1, 0].set_xlabel("Class Names")
axes[1, 0].set_ylabel("Number of Objects")
axes[1, 0].set_title("Object Counts per Class in Test Split")
axes[1, 0].tick_params(axis='x', rotation=45)
for bar in bars:
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(bar.get_height())), ha='center', va='bottom')

# Total number of objects in each dataset split
labels = ["Train", "Validation", "Test"]
object_counts = [sum(train_object_count.values()), sum(val_object_count.values()), sum(test_object_count.values())]
bars = axes[1, 1].bar(labels, object_counts, color=['blue', 'orange', 'green'])
axes[1, 1].set_xlabel("Dataset Split")
axes[1, 1].set_ylabel("Number of Objects")
axes[1, 1].set_title("Total Number of Objects in Each Dataset Split")
for idx, val in enumerate(object_counts):
    axes[1, 1].text(idx, val + 0.5, str(val), ha='center', va='bottom')

# Plot heatmap for Train split
sns.heatmap(heatmap_train, cmap="viridis", ax=axes[2, 0])
axes[2, 0].set_title("Average Object Locations - Train Split")
axes[2, 0].set_xlabel("X-axis (normalized)")
axes[2, 0].set_ylabel("Y-axis (normalized)")

# Plot heatmap for Validation split
sns.heatmap(heatmap_val, cmap="viridis", ax=axes[2, 1])
axes[2, 1].set_title("Average Object Locations - Validation Split")
axes[2, 1].set_xlabel("X-axis (normalized)")
axes[2, 1].set_ylabel("Y-axis (normalized)")

plt.tight_layout()
combined_plot_filename = os.path.join(output_dir, "combined_plots.png")
plt.savefig(combined_plot_filename)
plt.show()