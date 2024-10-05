"""""
Copyright (c) Mohamed Abdelkader 2024

This script is used to create a classification dataset from an instance segmentation dataset. The input dataset is expected to have the following structure:
- train, valid, test sub-directories, each containing images and labels sub-directories.
- The labels are .txt files containing the class ID and normalized coordinates of the mask points.

The script extracts bounding boxes around the object masks and saves them into a new classification directory, organized by class and split (train/valid/test).

Usage:
- Pass the main dataset directory path as an argument.
- Optionally, provide resize dimensions for the cropped images.

Example command:
python script_name.py /path/to/dataset --resize 224 224
"""

import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm


def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def extract_bounding_box(points, img_width, img_height):
    x_coords = [int(float(x) * img_width) for x, y in points]
    y_coords = [int(float(y) * img_height) for x, y in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max


def crop_and_save_image(img, bbox, class_name, output_dir, img_name, resize_dim=None):
    x_min, y_min, x_max, y_max = bbox
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    if cropped_img.size == 0:
        return
    
    if resize_dim:
        cropped_img = cv2.resize(cropped_img, resize_dim)
    
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    output_path = os.path.join(class_output_dir, img_name)
    cv2.imwrite(output_path, cropped_img)


def process_dataset(dataset_dir, resize_dim):
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    data_info = load_yaml_file(data_yaml_path)
    
    splits = ['train', 'valid', 'test']
    classes = data_info['names']
    classification_dir = os.path.join(dataset_dir, 'classification')
    os.makedirs(classification_dir, exist_ok=True)

    for split in splits:
        split_images_path = os.path.join(dataset_dir, split, 'images')
        split_labels_path = os.path.join(dataset_dir, split, 'labels')
        split_output_dir = os.path.join(classification_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(split_images_path), desc=f'Processing {split} split'):
            if img_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(split_images_path, img_name)
                label_path = os.path.join(split_labels_path, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

                if not os.path.exists(label_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]

                with open(label_path, 'r') as label_file:
                    for line in label_file:
                        line = line.strip().split()
                        class_id = int(line[0])
                        points = [(line[i], line[i + 1]) for i in range(1, len(line), 2)]
                        
                        bbox = extract_bounding_box(points, img_width, img_height)
                        crop_and_save_image(img, bbox, classes[class_id], split_output_dir, img_name, resize_dim)


def main():
    parser = argparse.ArgumentParser(description='Create classification dataset from instance segmentation dataset')
    parser.add_argument('dataset_dir', type=str, help='Path to the main dataset directory')
    parser.add_argument('--resize', type=int, nargs=2, help='Resize dimensions for cropped images (width height)')
    
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    resize_dim = tuple(args.resize) if args.resize else None

    process_dataset(dataset_dir, resize_dim)


if __name__ == "__main__":
    main()