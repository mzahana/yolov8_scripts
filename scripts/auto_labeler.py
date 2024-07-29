"""
Auto Labeler Script for YOLOv8

This script performs inference on a set of images using a YOLOv8 model to detect objects and segment them. 
It then saves the images and their corresponding labels into different directories based on the detected classes.

Functionality:
1. Loads a YOLOv8 model and performs segmentation on images.
2. Categorizes images into 'bundles' and 'single' based on detected object classes:
   - 'Bundle' class (0) images are saved in the 'bundles' directory.
   - 'Single-Item' class (1) images are saved in the 'single' directory.
3. Optionally, saves images with drawn masks indicating detected objects.
4. Simplifies the segmentation mask polygons to reduce the number of points while preserving the shape.
5. Outputs the processed images and corresponding label files in the respective directories.

Usage:
    python3 auto_labeler.py <image_dir> <model_path> [--save-masked-images] [--epsilon <value>]

Arguments:
    image_dir: Path to the directory containing images (PNG or JPG format).
    model_path: Path to the YOLOv8 .pt model file.
    --save-masked-images: Optional flag to save images with drawn masks.
    --epsilon: Optional value for polygon simplification (default: 0.01).

Example:
    python3 auto_labeler.py /path/to/images /path/to/model.pt --save-masked-images --epsilon 0.02
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
import numpy as np

class YOLOInference:
    def __init__(self, model_path, image_dir, confidence=0.5, save_masked_images=False, epsilon=0.01):
        self.model_path_ = model_path
        self.image_dir_ = Path(image_dir)
        self.confidence_ = confidence
        self.save_masked_images_ = save_masked_images
        self.epsilon_ = epsilon

        # Define parent directory and output directories for images and labels
        self.parent_dir_ = self.image_dir_.parent
        self.bundle_images_dir_ = self.parent_dir_ / "bundles/images"
        self.bundle_labels_dir_ = self.parent_dir_ / "bundles/labels"
        self.single_images_dir_ = self.parent_dir_ / "single/images"
        self.single_labels_dir_ = self.parent_dir_ / "single/labels"
        self.masked_images_dir_ = self.parent_dir_ / "masked_images"
        
        # Ensure all output directories exist
        for dir in [self.bundle_images_dir_, self.bundle_labels_dir_, self.single_images_dir_, self.single_labels_dir_, self.masked_images_dir_]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Load model with explicit task definition
        self.model_ = YOLO(self.model_path_, task='segment')
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def draw_masks(self, image, masks, classes):
        for mask, cls in zip(masks, classes):
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
        return image

    def simplify_contour(self, contour):
        epsilon = self.epsilon_ * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def run_inference(self):
        total_images = 0
        bundle_count = 0
        single_count = 0

        # Iterate over images in the image directory
        image_paths = list(self.image_dir_.glob('*.png')) + list(self.image_dir_.glob('*.jpg'))
        total_images = len(image_paths)
        
        for img_file in tqdm(image_paths, desc="Processing Images"):
            try:
                # Perform inference on each image
                results = self.model_.predict(
                    source=str(img_file),
                    verbose=False,
                    stream=False,
                    conf=self.confidence_,
                    device=self.device_
                )
            except Exception as e:
                print(f"WARNING ⚠️ Image Read Error {img_file}: {e}")
                continue
            
            results = results[0].cpu()
            contains_bundle = False
            all_single = True
            label_data = []

            if results.masks and results.boxes:
                for box_data, mask_data in zip(results.boxes, results.masks.data):
                    class_id = int(box_data.cls)
                    if class_id == 0:  # Bundle class
                        contains_bundle = True
                        all_single = False
                    elif class_id == 1:  # Single-Item class
                        all_single = True
                    
                    # Extract the segmentation mask points
                    mask = mask_data.cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (results.orig_shape[1], results.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        simplified_contour = self.simplify_contour(contour)
                        if len(simplified_contour) < 3:
                            continue  # Skip contours that are too small to be valid polygons
                        normalized_points = [(point[0][0] / results.orig_shape[1], point[0][1] / results.orig_shape[0]) for point in simplified_contour]
                        label_data.append((class_id, normalized_points))
                
                # Prepare label content
                label_file_content = ""
                for class_id, points in label_data:
                    points_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                    label_file_content += f"{class_id} {points_str}\n"
                
                # Determine the corresponding label file
                label_file_path = img_file.with_suffix('.txt')
                
                if contains_bundle:
                    # Save image and label in the bundle directories
                    shutil.copy(img_file, self.bundle_images_dir_)
                    bundle_count += 1
                    with open(self.bundle_labels_dir_ / label_file_path.name, 'w') as f:
                        f.write(label_file_content)
                elif all_single:
                    # Save image and label in the single directories
                    shutil.copy(img_file, self.single_images_dir_)
                    single_count += 1
                    with open(self.single_labels_dir_ / label_file_path.name, 'w') as f:
                        f.write(label_file_content)

                # Optionally save the masked image
                if self.save_masked_images_:
                    img = cv2.imread(str(img_file))
                    masked_img = self.draw_masks(img, results.masks.data, results.boxes.cls)
                    masked_img_path = self.masked_images_dir_ / img_file.name
                    cv2.imwrite(str(masked_img_path), masked_img)

        # Print the summary
        print(f"Total images processed: {total_images}")
        print(f"Images saved in 'bundles': {bundle_count}")
        print(f"Images saved in 'single': {single_count}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images")
    parser.add_argument("model_path", type=str, help="Path to the YOLOv8 .pt model")
    parser.add_argument("--save-masked-images", action="store_true", help="Save images with drawn masks")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon value for polygon simplification")

    args = parser.parse_args()

    yolo_inference = YOLOInference(
        model_path=args.model_path,
        image_dir=args.image_dir,
        confidence=0.5,
        save_masked_images=args.save_masked_images,
        epsilon=args.epsilon
    )

    yolo_inference.run_inference()
