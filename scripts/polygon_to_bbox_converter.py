"""
This script converts polygon annotations in label files to bounding boxes and optionally visualizes 
the results by overlaying the bounding boxes on the corresponding images. The converted bounding 
boxes are saved in a new directory, and a single combined image showing a random selection of images 
with their bounding boxes is saved for visual verification.

The input label files should be in the format:
    class_id x1 y1 x2 y2 ...

The output label files will be in the format:
    class_id center_x center_y width height

Dependencies:
- numpy
- argparse
- random
- matplotlib
- tqdm

Example usage:
--------------
To convert polygon annotations to bounding boxes and visualize the result:

    python polygon_to_bbox_converter.py /path/to/labels/folder --images_folder /path/to/images/folder --num_images 6

Parameters:
-----------
- labels_folder: Path to the folder containing the label .txt files with polygon annotations.
- images_folder (optional): Path to the folder containing the corresponding images for visualization.
- num_images (optional): Number of random images to visualize with bounding boxes (default: 6).
"""
import os
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
from matplotlib.image import imread

def convert_polygon_to_bbox(polygon_coords):
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

def process_labels(labels_folder, images_folder=None, num_images=6):
    # Create the output directory if it doesn't exist
    output_folder = os.path.join(os.path.dirname(labels_folder), 'labels_boxes')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all .txt files in the labels folder
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    total_files = len(label_files)
    
    # Process each file
    for label_file in tqdm(label_files, desc="Processing labels"):
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
    
    print(f"Total number of files processed: {total_files}")

    # If images folder is provided, visualize and save all random images in a single plot
    if images_folder:
        save_combined_image(images_folder, output_folder, label_files, num_images, labels_folder)

def save_combined_image(images_folder, labels_folder, label_files, num_images, labels_folder_parent):
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
    
    combined_image_path = os.path.join(os.path.dirname(labels_folder), 'combined_bbox_image.png')
    plt.savefig(combined_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Convert polygon annotations to bounding boxes with optional visual verification.")
    parser.add_argument('labels_folder', type=str, help="Path to the folder containing label .txt files")
    parser.add_argument('--images_folder', type=str, help="Path to the folder containing images for visual verification", default=None)
    parser.add_argument('--num_images', type=int, help="Number of random images to visualize with bounding boxes", default=6)
    args = parser.parse_args()
    
    process_labels(args.labels_folder, args.images_folder, args.num_images)

if __name__ == "__main__":
    main()
