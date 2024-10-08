"""
Copyright (c) Mohamed Abdelkader 2024

This script performs data augmentation for instance segmentation tasks. It extracts objects based on polygon annotations
from label files and places them on a specified background image, applying various augmentations such as rotation, blurring,
scaling, and contrast adjustment.

Functionality:
- Extract objects from images based on their polygon annotations in the corresponding label files.
- Apply a series of augmentations to the extracted objects, including:
  - Rotation: Rotates the object by a random angle within a specified range.
  - Blurring: Applies Gaussian blur to the object with a randomly chosen intensity.
  - Scaling: Scales the object randomly within a specified range.
  - Contrast Adjustment: Adjusts the object's contrast randomly within a specified range.
- Place the augmented objects onto a specified background image in random locations, with a defined region scale.
- Save the resulting augmented images and their corresponding updated label files.

Usage:
- Provide the path to the image directory containing `images` and `labels` subdirectories.
- Specify the class ID of the object to be augmented, the background image path, and the output directory.
- Optional arguments allow customization of the number of augmentations, rotation range, blur range, scaling range, 
  contrast range, and region scale.

Note:
- At least one augmentation type must be specified, otherwise the script will print an error and exit.

Example:
    python augment_instance.py /path/to/data 1 /path/to/background.jpg /path/to/output 
    --num_augmentations 5 --rotation_range -20 20 --blur_range 0 3 --scaling_range 0.7 1.3 --contrast_range 0.8 1.2

"""

import os
import cv2
import numpy as np
import random
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_label_file(label_file, images_dir, labels_dir, background_img, output_dir, class_id, num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, region_scale, image_w, image_h, max_region_w, max_region_h):
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as lf:
        lines = lf.readlines()

    generated_objects_count = 0
    results = []

    for line in lines:
        parts = line.strip().split()
        obj_class_id = parts[0]

        # Process only the specified class ID
        if obj_class_id != str(class_id):
            continue

        # Parse the polygon points (x, y)
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        # Denormalize the coordinates
        coords[:, 0] *= image_w
        coords[:, 1] *= image_h

        # Extract object mask and ROI
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        polygon = np.array(coords, np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        obj_roi = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(polygon)
        obj_roi = obj_roi[y:y + h, x:x + w]
        mask_roi = mask[y:y + h, x:x + w]

        # Augment the object
        for i in range(num_augmentations):
            aug_obj = obj_roi.copy()
            aug_mask = mask_roi.copy()

            # Rotation
            if rotation_range:
                angle = random.uniform(*rotation_range)
                matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                aug_obj = cv2.warpAffine(aug_obj, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                aug_mask = cv2.warpAffine(aug_mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Blurring
            if blur_range:
                blur_value = random.randint(*blur_range)
                if blur_value > 0:
                    aug_obj = cv2.GaussianBlur(aug_obj, (blur_value * 2 + 1, blur_value * 2 + 1), 0)

            # Scaling
            if scaling_range:
                scale_factor = random.uniform(*scaling_range)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                new_w, new_h = w, h

            # Contrast Adjustment
            if contrast_range:
                alpha = random.uniform(*contrast_range)
                aug_obj = cv2.convertScaleAbs(aug_obj, alpha=alpha, beta=0)

            # Ensure the object fits within the defined region scale
            new_w = min(new_w, max_region_w)
            new_h = min(new_h, max_region_h)
            aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Adjust position to ensure better distribution across the entire region
            rand_x = random.randint(0, max(image_w - new_w, 0))
            rand_y = random.randint(0, max(image_h - new_h, 0))

            # Place the augmented object on the background
            bg_copy = background_img.copy()

            # Blend the augmented object with the background using the mask
            for c in range(3):  # Iterate over each color channel
                bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] = (
                    bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] * (1 - aug_mask / 255) +
                    aug_obj[:, :, c] * (aug_mask / 255)
                )

            # Save augmented image and label
            aug_image_name = f"{image_name.replace('.jpg', '')}_aug_{i}.jpg"
            aug_label_name = f"{label_file.replace('.txt', '')}_aug_{i}.txt"
            cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
            with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                new_coords = (coords - [x, y]) * (new_w / w, new_h / h)
                new_coords[:, 0] = (new_coords[:, 0] + rand_x) / image_w
                new_coords[:, 1] = (new_coords[:, 1] + rand_y) / image_h
                new_coords = new_coords.reshape(-1)
                lf_aug.write(f"{class_id} {' '.join(map(str, new_coords))}\n")

            generated_objects_count += 1

    return generated_objects_count

def augment_instance(image_dir, class_id, background_img_path, output_dir=None, 
                     num_augmentations=3, rotation_range=None, blur_range=None, 
                     scaling_range=None, contrast_range=None, region_scale=0.8):
    # Ensure at least one augmentation type is specified
    if not any([rotation_range, blur_range, scaling_range, contrast_range]):
        print("Error: At least one augmentation type must be specified.")
        sys.exit(1)

    # Prepare directories
    images_dir = os.path.join(image_dir, 'images')
    labels_dir = os.path.join(image_dir, 'labels')
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'augmented')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'images')):
        os.makedirs(os.path.join(output_dir, 'images'))
    if not os.path.exists(os.path.join(output_dir, 'labels')):
        os.makedirs(os.path.join(output_dir, 'labels'))

    # Load the background image
    background_img = cv2.imread(background_img_path)

    # Get the size of the images in the dataset (assuming all are the same size)
    sample_image_path = os.path.join(images_dir, os.listdir(images_dir)[0])
    sample_image = cv2.imread(sample_image_path)
    image_h, image_w = sample_image.shape[:2]

    # Resize the background image to match the size of the dataset images
    background_img = cv2.resize(background_img, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    max_region_h, max_region_w = int(region_scale * image_h), int(region_scale * image_w)

    # Iterate over label files in parallel
    label_files = os.listdir(labels_dir)
    total_labels = len(label_files)
    generated_objects_count = 0

    with tqdm(total=total_labels, desc="Processing Label Files") as progress_bar:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_label_file, label_file, images_dir, labels_dir, background_img, output_dir, class_id,
                    num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, region_scale,
                    image_w, image_h, max_region_w, max_region_h
                ) for label_file in label_files
            ]

            for future in as_completed(futures):
                generated_objects_count += future.result()
                progress_bar.update(1)

    print(f"Total generated objects after augmentation: {generated_objects_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Instance Segmentation Data Augmentation')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing images and labels subdirectories')
    parser.add_argument('class_id', type=int, help='Class ID to augment')
    parser.add_argument('background_img_path', type=str, help='Path to the background image')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, help='Directory to save the augmented images and labels')
    parser.add_argument('--num_augmentations', type=int, default=3, help='Number of augmentations per object')
    parser.add_argument('--rotation_range', type=float, nargs=2, help='Range of rotation angles in degrees')
    parser.add_argument('--blur_range', type=int, nargs=2, help='Range of blurring kernel size')
    parser.add_argument('--scaling_range', type=float, nargs=2, help='Range of scaling factors')
    parser.add_argument('--contrast_range', type=float, nargs=2, help='Range of contrast adjustment factors')
    parser.add_argument('--region_scale', type=float, default=0.8, help='Scale of the region to place objects within the background')

    args = parser.parse_args()
    augment_instance(args.image_dir, args.class_id, args.background_img_path, args.output_dir,
                     args.num_augmentations, args.rotation_range, args.blur_range,
                     args.scaling_range, args.contrast_range, args.region_scale)