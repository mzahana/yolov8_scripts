import os
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
from matplotlib.image import imread

def convert_polygon_to_bbox(polygon_coords):
    """
    Convert polygon coordinates to bounding box coordinates.
    Args:
        polygon_coords (list): List of polygon coordinates in the format [x1, y1, x2, y2, ...].
    Returns:
        list: Bounding box coordinates in the format [center_x, center_y, width, height].
    """
    x_coords = polygon_coords[0::2]
    y_coords = polygon_coords[1::2]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # Calculate center, width, and height
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    return [center_x, center_y, width, height]

def process_labels(labels_folder, output_folder):
    """
    Process all label files in the given folder and convert polygon annotations to bounding boxes.
    Args:
        labels_folder (str): Path to the folder containing label files.
        output_folder (str): Path to the folder where converted label files will be saved.
    """
    # Get all .txt files in the labels folder
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    # Process each file
    for label_file in tqdm(label_files, desc=f"Processing {os.path.basename(labels_folder)}"):
        input_file_path = os.path.join(labels_folder, label_file)
        output_file_path = os.path.join(output_folder, label_file)
        
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                elements = line.strip().split()
                label_id = elements[0]
                polygon_coords = list(map(float, elements[1:]))
                bbox = convert_polygon_to_bbox(polygon_coords)
                bbox_str = ' '.join(map(str, bbox))
                outfile.write(f"{label_id} {bbox_str}\n")

def save_combined_image(images_folder, labels_folder, label_files, num_images, output_path):
    """
    Save a combined image with bounding boxes overlaid on randomly selected images.
    Args:
        images_folder (str): Path to the folder containing images.
        labels_folder (str): Path to the folder containing bounding box labels.
        label_files (list): List of label file names.
        num_images (int): Number of random images to visualize.
        output_path (str): Path to save the combined visualization image.
    """
    selected_files = random.sample(label_files, min(num_images, len(label_files)))
    images = []

    fig, axs = plt.subplots(1, len(selected_files), figsize=(15, 6))

    if len(selected_files) == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one subplot

    for i, label_file in enumerate(selected_files):
        image_name = label_file.replace('.txt', '.png')
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            image_name = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found for {label_file}, skipping.")
            continue

        # Load image using Matplotlib
        image = imread(image_path)
        h, w, _ = image.shape

        # Load corresponding bounding box file
        bbox_file_path = os.path.join(labels_folder, label_file)
        with open(bbox_file_path, 'r') as bbox_file:
            for line in bbox_file:
                elements = line.strip().split()
                label_id = elements[0]
                bbox = list(map(float, elements[1:]))

                # Denormalize coordinates for plotting
                center_x = bbox[0] * w
                center_y = bbox[1] * h
                width = bbox[2] * w
                height = bbox[3] * h

                x_min = center_x - width / 2
                y_min = center_y - height / 2

                # Draw bounding box on the image
                rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='green', facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(x_min, y_min - 10, label_id, color='green', fontsize=12, weight='bold')

        axs[i].imshow(image)
        axs[i].axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_dataset(parent_dir, num_images=6):
    """
    Process the entire dataset, including train, valid, and test subdirectories.
    Args:
        parent_dir (str): Path to the parent directory containing train, valid, and test subdirectories.
        num_images (int): Number of random images to visualize with bounding boxes.
    """
    # Create the output directory with _bbx suffix
    output_parent_dir = parent_dir + "_bbx"
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)
    
    # Process train, valid, and test subdirectories
    for subdir in ['train', 'valid', 'test']:
        input_subdir = os.path.join(parent_dir, subdir)
        output_subdir = os.path.join(output_parent_dir, subdir)
        
        if not os.path.exists(input_subdir):
            print(f"Skipping {subdir} as it does not exist.")
            continue
        
        # Create output subdirectories for images and labels
        output_images_dir = os.path.join(output_subdir, 'images')
        output_labels_dir = os.path.join(output_subdir, 'labels')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        # Copy images to the output directory (no changes needed for images)
        input_images_dir = os.path.join(input_subdir, 'images')
        for image_file in os.listdir(input_images_dir):
            if image_file.endswith(('.jpg', '.png')):
                os.system(f"cp {os.path.join(input_images_dir, image_file)} {output_images_dir}")
        
        # Process labels
        input_labels_dir = os.path.join(input_subdir, 'labels')
        process_labels(input_labels_dir, output_labels_dir)
    
    # Save a combined visualization of random images with bounding boxes
    if num_images > 0:
        train_images_dir = os.path.join(parent_dir, 'train', 'images')
        train_labels_dir = os.path.join(output_parent_dir, 'train', 'labels')
        train_label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        combined_image_path = os.path.join(output_parent_dir, 'combined_bbox_visualization.png')
        save_combined_image(train_images_dir, train_labels_dir, train_label_files, num_images, combined_image_path)
        print(f"Combined visualization saved to {combined_image_path}")
    
    print(f"Dataset processing complete. Output saved to {output_parent_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert polygon annotations to bounding boxes for a dataset.")
    parser.add_argument('parent_dir', type=str, help="Path to the parent directory containing train, valid, and test subdirectories.")
    parser.add_argument('--num_images', type=int, help="Number of random images to visualize with bounding boxes", default=6)
    args = parser.parse_args()
    
    process_dataset(args.parent_dir, args.num_images)

if __name__ == "__main__":
    main()