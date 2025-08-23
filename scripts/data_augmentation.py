"""
Copyright (c) Mohamed Abdelkader 2024 - Modified for multi-class support
Fixed version with proper coordinate transformation for rotation

This script performs data augmentation for instance segmentation tasks. It extracts objects based on polygon annotations
from label files and places them on a specified background image, applying various augmentations such as rotation, blurring,
scaling, and contrast adjustment.

Functionality:
- Extract objects from images based on their polygon annotations in the corresponding label files.
- Support multiple class IDs for augmentation
- Apply a series of augmentations to the extracted objects, including:
  - Rotation: Rotates the object by a random angle within a specified range.
  - Blurring: Applies Gaussian blur to the object with a randomly chosen intensity.
  - Scaling: Scales the object randomly within a specified range.
  - Contrast Adjustment: Adjusts the object's contrast randomly within a specified range.
- Place the augmented objects onto a specified background image in random locations, with a defined region scale.
- Save the resulting augmented images and their corresponding updated label files.

Usage:
- Provide the path to the image directory containing `images` and `labels` subdirectories.
- Specify the class IDs of the objects to be augmented (space-separated), the background image path, and the output directory.
- Optional arguments allow customization of the number of augmentations, rotation range, blur range, scaling range, 
  contrast range, and region scale.

Note:
- At least one augmentation type must be specified, otherwise the script will print an error and exit.

Example:
    python augment_instance.py /path/to/data 1 2 3 /path/to/background.jpg /path/to/output 
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

def transform_coordinates(coords, rotation_matrix, scale_factor, translation):
    """Transform coordinates through rotation, scaling, and translation"""
    transformed_coords = []
    
    for coord in coords:
        # Apply rotation
        if rotation_matrix is not None:
            # Convert to homogeneous coordinates
            point = np.array([coord[0], coord[1], 1])
            rotated_point = rotation_matrix @ point
            coord = [rotated_point[0], rotated_point[1]]
        
        # Apply scaling
        if scale_factor != 1.0:
            coord = [coord[0] * scale_factor, coord[1] * scale_factor]
        
        # Apply translation
        coord = [coord[0] + translation[0], coord[1] + translation[1]]
        
        transformed_coords.append(coord)
    
    return np.array(transformed_coords)

def process_label_file(label_file, images_dir, labels_dir, background_img, output_dir, class_ids, num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, region_scale, image_w, image_h, max_region_w, max_region_h, augment_together=False):
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as lf:
        lines = lf.readlines()

    generated_objects_count = 0

    # Parse all objects of specified classes from the label file
    valid_objects = []
    for line in lines:
        parts = line.strip().split()
        obj_class_id = int(parts[0])

        # Process only the specified class IDs
        if obj_class_id not in class_ids:
            continue

        # Parse the polygon points (x, y)
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        valid_objects.append((obj_class_id, coords))

    if not valid_objects:
        return 0

    image_name = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)

    if augment_together:
        # Augment all valid objects together on the same background
        for i in range(num_augmentations):
            bg_copy = background_img.copy()
            aug_labels = []
            
            for obj_class_id, coords in valid_objects:
                # Process each object
                coords_denorm = coords.copy()
                coords_denorm[:, 0] *= image_w
                coords_denorm[:, 1] *= image_h

                # Extract object mask and ROI
                mask = np.zeros((image_h, image_w), dtype=np.uint8)
                polygon = np.array(coords_denorm, np.int32)
                cv2.fillPoly(mask, [polygon], 255)
                obj_roi = cv2.bitwise_and(image, image, mask=mask)
                x, y, w, h = cv2.boundingRect(polygon)
                obj_roi = obj_roi[y:y + h, x:x + w]
                mask_roi = mask[y:y + h, x:x + w]

                # Get coordinates relative to ROI
                coords_relative = coords_denorm - [x, y]

                # Apply augmentations and get transformation parameters
                aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = apply_augmentations(
                    obj_roi, mask_roi, w, h, rotation_range, blur_range, 
                    scaling_range, contrast_range, max_region_w, max_region_h
                )

                # Skip if augmentation failed
                if aug_obj is None or new_w <= 0 or new_h <= 0:
                    continue

                # Place object on background
                rand_x = random.randint(0, max(image_w - new_w, 0))
                rand_y = random.randint(0, max(image_h - new_h, 0))

                # Blend the augmented object with the background using the mask
                for c in range(3):  # Iterate over each color channel
                    bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] = (
                        bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] * (1 - aug_mask / 255) +
                        aug_obj[:, :, c] * (aug_mask / 255)
                    )

                # Transform coordinates properly
                translation = [rand_x, rand_y]
                new_coords = transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                
                # Normalize coordinates
                new_coords[:, 0] /= image_w
                new_coords[:, 1] /= image_h
                new_coords = new_coords.reshape(-1)
                aug_labels.append(f"{obj_class_id} {' '.join(map(str, new_coords))}")

            # Save augmented image and labels
            aug_image_name = f"{image_name.replace('.jpg', '')}_multi_aug_{i}.jpg"
            aug_label_name = f"{label_file.replace('.txt', '')}_multi_aug_{i}.txt"
            cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
            with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                for label in aug_labels:
                    lf_aug.write(f"{label}\n")

            generated_objects_count += len(valid_objects)

    else:
        # Augment each object separately
        for obj_class_id, coords in valid_objects:
            # Denormalize the coordinates
            coords_denorm = coords.copy()
            coords_denorm[:, 0] *= image_w
            coords_denorm[:, 1] *= image_h

            # Extract object mask and ROI
            mask = np.zeros((image_h, image_w), dtype=np.uint8)
            polygon = np.array(coords_denorm, np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            obj_roi = cv2.bitwise_and(image, image, mask=mask)
            x, y, w, h = cv2.boundingRect(polygon)
            obj_roi = obj_roi[y:y + h, x:x + w]
            mask_roi = mask[y:y + h, x:x + w]

            # Get coordinates relative to ROI
            coords_relative = coords_denorm - [x, y]

            # Augment the object multiple times
            for i in range(num_augmentations):
                # Apply augmentations and get transformation parameters
                aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = apply_augmentations(
                    obj_roi, mask_roi, w, h, rotation_range, blur_range, 
                    scaling_range, contrast_range, max_region_w, max_region_h
                )

                # Skip if augmentation failed
                if aug_obj is None or new_w <= 0 or new_h <= 0:
                    continue

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

                # Transform coordinates properly
                translation = [rand_x, rand_y]
                new_coords = transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                
                # Normalize coordinates
                new_coords[:, 0] /= image_w
                new_coords[:, 1] /= image_h
                new_coords = new_coords.reshape(-1)

                # Save augmented image and label
                aug_image_name = f"{image_name.replace('.jpg', '')}_class_{obj_class_id}_aug_{i}.jpg"
                aug_label_name = f"{label_file.replace('.txt', '')}_class_{obj_class_id}_aug_{i}.txt"
                cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
                with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                    lf_aug.write(f"{obj_class_id} {' '.join(map(str, new_coords))}\n")

                generated_objects_count += 1

    return generated_objects_count

def apply_augmentations(obj_roi, mask_roi, w, h, rotation_range, blur_range, scaling_range, contrast_range, max_region_w, max_region_h):
    """Apply augmentations to an object and return the augmented object, mask, new dimensions, and transformation parameters"""
    aug_obj = obj_roi.copy()
    aug_mask = mask_roi.copy()
    rotation_matrix = None
    scale_factor = 1.0

    # Check for minimum valid dimensions
    if w <= 0 or h <= 0:
        print(f"Warning: Invalid object dimensions (w={w}, h={h}). Skipping augmentation.")
        return None, None, 0, 0, None, 1.0

    # Rotation
    if rotation_range:
        angle = random.uniform(*rotation_range)
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug_obj = cv2.warpAffine(aug_obj, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        aug_mask = cv2.warpAffine(aug_mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Blurring
    if blur_range:
        blur_value = random.randint(*blur_range)
        if blur_value > 0:
            aug_obj = cv2.GaussianBlur(aug_obj, (blur_value * 2 + 1, blur_value * 2 + 1), 0)

    # Scaling
    if scaling_range:
        scale_factor = random.uniform(*scaling_range)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        # Ensure minimum dimensions after scaling
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Check if scaling would make object too small
        if new_w > 0 and new_h > 0:
            aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            new_w, new_h = w, h
            scale_factor = 1.0
    else:
        new_w, new_h = w, h

    # Contrast Adjustment
    if contrast_range:
        alpha = random.uniform(*contrast_range)
        aug_obj = cv2.convertScaleAbs(aug_obj, alpha=alpha, beta=0)

    # Ensure the object fits within the defined region scale
    if max_region_w > 0 and max_region_h > 0:
        if new_w > max_region_w or new_h > max_region_h:
            # Scale down to fit within region constraints
            w_ratio = max_region_w / new_w if new_w > max_region_w else 1.0
            h_ratio = max_region_h / new_h if new_h > max_region_h else 1.0
            ratio = min(w_ratio, h_ratio)
            
            new_w = max(1, int(new_w * ratio))
            new_h = max(1, int(new_h * ratio))
            
            if new_w > 0 and new_h > 0:
                aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_factor *= ratio
            else:
                # If still invalid, return original dimensions
                new_w, new_h = w, h
                aug_obj = obj_roi.copy()
                aug_mask = mask_roi.copy()
                scale_factor = 1.0

    # Final check for valid dimensions
    if new_w <= 0 or new_h <= 0:
        print(f"Warning: Final dimensions invalid (new_w={new_w}, new_h={new_h}). Using original dimensions.")
        new_w, new_h = w, h
        aug_obj = obj_roi.copy()
        aug_mask = mask_roi.copy()
        scale_factor = 1.0

    return aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor

def augment_instance(image_dir, class_ids, background_img_path, output_dir=None, 
                     num_augmentations=3, rotation_range=None, blur_range=None, 
                     scaling_range=None, contrast_range=None, region_scale=0.8, 
                     augment_together=False):
    # Ensure at least one augmentation type is specified
    if not any([rotation_range, blur_range, scaling_range, contrast_range]):
        print("Error: At least one augmentation type must be specified.")
        sys.exit(1)

    # Convert class_ids to list if it's a single integer
    if isinstance(class_ids, int):
        class_ids = [class_ids]

    print(f"Augmenting classes: {class_ids}")
    print(f"Augment together: {augment_together}")

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
                    process_label_file, label_file, images_dir, labels_dir, background_img, output_dir, class_ids,
                    num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, region_scale,
                    image_w, image_h, max_region_w, max_region_h, augment_together
                ) for label_file in label_files
            ]

            for future in as_completed(futures):
                generated_objects_count += future.result()
                progress_bar.update(1)

    print(f"Total generated objects after augmentation: {generated_objects_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Class Instance Segmentation Data Augmentation')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing images and labels subdirectories')
    parser.add_argument('class_ids', type=int, nargs='+', help='Class IDs to augment (space-separated)')
    parser.add_argument('background_img_path', type=str, help='Path to the background image')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the augmented images and labels')
    parser.add_argument('--num_augmentations', type=int, default=3, help='Number of augmentations per object')
    parser.add_argument('--rotation_range', type=float, nargs=2, help='Range of rotation angles in degrees')
    parser.add_argument('--blur_range', type=int, nargs=2, help='Range of blurring kernel size')
    parser.add_argument('--scaling_range', type=float, nargs=2, help='Range of scaling factors')
    parser.add_argument('--contrast_range', type=float, nargs=2, help='Range of contrast adjustment factors')
    parser.add_argument('--region_scale', type=float, default=0.8, help='Scale of the region to place objects within the background')
    parser.add_argument('--augment_together', action='store_true', help='Augment all specified classes together on the same background')

    args = parser.parse_args()
    augment_instance(args.image_dir, args.class_ids, args.background_img_path, args.output_dir,
                     args.num_augmentations, args.rotation_range, args.blur_range,
                     args.scaling_range, args.contrast_range, args.region_scale, 
                     args.augment_together)