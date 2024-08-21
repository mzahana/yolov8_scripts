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
    return [min_x, min_y, max_x, max_y]

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

                # Denormalize coordinates
                x_min = bbox[0] * w
                y_min = bbox[1] * h
                x_max = bbox[2] * w
                y_max = bbox[3] * h

                # Draw bounding box on the image
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='green', facecolor='none')
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
