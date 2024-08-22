"""
Auto Labeler Script for YOLOv8

This script performs inference on a set of images using a YOLOv8 model to detect objects and segment them. 
It then saves the images and their corresponding labels into different directories based on the detected classes.

Functionality:
1. Loads a YOLOv8 model and performs segmentation or detection on images.
2. Categorizes images into 'bundles' and 'single' based on detected object classes:
   - 'Bundle' class (0) images are saved in the 'bundles' directory.
   - 'Single-Item' class (1) images are saved in the 'single' directory.
3. Optionally, saves images with drawn masks or bounding boxes indicating detected objects.
4. If masks are available, simplifies the segmentation mask polygons to reduce the number of points while preserving the shape.
5. If masks are not available, generates bounding box labels in the YOLOv8 format.
6. Outputs the processed images and corresponding label files in the respective directories.
7. Supports reading .png, .jpg, and .tif images.
8. Allows optional resizing of images to specified dimensions before passing them to the model.
9. Always saves images in the individual folders as `.jpg`.

Usage:
    python3 auto_labeler.py <image_dir> <model_path> [--save-masked-images] [--epsilon <value>] [--resize-height <height>] [--resize-width <width>]

Arguments:
    image_dir: Path to the directory containing images (PNG, JPG, or TIF format).
    model_path: Path to the YOLOv8 .pt model file.
    --save-masked-images: Optional flag to save images with drawn masks or bounding boxes.
    --epsilon: Optional value for polygon simplification (default: 0.01).
    --resize-height: Optional height to resize the images before passing them to the model.
    --resize-width: Optional width to resize the images before passing them to the model.

Example:
    python3 auto_labeler.py /path/to/images /path/to/model.pt --save-masked-images --epsilon 0.02 --resize-height 640 --resize-width 480
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
    def __init__(self, model_path, image_dir, confidence=0.5, save_masked_images=False, epsilon=0.01, resize_height=None, resize_width=None):
        self.model_path_ = model_path
        self.image_dir_ = Path(image_dir)
        self.confidence_ = confidence
        self.save_masked_images_ = save_masked_images
        self.epsilon_ = epsilon
        self.resize_height_ = resize_height
        self.resize_width_ = resize_width

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
        self.model_ = YOLO(self.model_path_, task='segment' if 'segment' in self.model_path_ else 'detect')
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def draw_masks(self, image, masks, classes):
        for mask, cls in zip(masks, classes):
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
        return image

    def draw_bounding_boxes(self, image, boxes, classes):
        for box, cls in zip(boxes, classes):
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
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
        image_paths = list(self.image_dir_.glob('*.png')) + list(self.image_dir_.glob('*.jpg')) + list(self.image_dir_.glob('*.tif'))
        total_images = len(image_paths)
        
        for img_file in tqdm(image_paths, desc="Processing Images"):
            try:
                # Read the image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"WARNING ⚠️ Unable to read image {img_file}")
                    continue

                # Optionally resize the image
                if self.resize_height_ is not None and self.resize_width_ is not None:
                    img = cv2.resize(img, (self.resize_width_, self.resize_height_))

                # Perform inference on each image
                results = self.model_.predict(
                    source=img,
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

            # If masks are available, generate mask labels
            if hasattr(results, 'masks') and results.masks:
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

            # If no masks, generate bounding box labels
            elif hasattr(results, 'boxes') and results.boxes:
                for box_data in results.boxes:
                    class_id = int(box_data.cls)
                    if class_id == 0:  # Bundle class
                        contains_bundle = True
                        all_single = False
                    elif class_id == 1:  # Single-Item class
                        all_single = True

                    # YOLOv8 format: class_id center_x center_y width height (normalized)
                    x1, y1, x2, y2 = box_data.xyxy[0]
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2
                    normalized_bbox = [
                        class_id,
                        center_x / img.shape[1],
                        center_y / img.shape[0],
                        width / img.shape[1],
                        height / img.shape[0]
                    ]
                    label_data.append(normalized_bbox)

            # Prepare label content
            label_file_content = ""
            if hasattr(results, 'masks') and results.masks:
                for class_id, points in label_data:
                    points_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                    label_file_content += f"{int(class_id)} {points_str}\n"
            else:
                for bbox in label_data:
                    label_file_content += f"{int(bbox[0])} " + " ".join(f"{x:.6f}" for x in bbox[1:]) + "\n"
                
            # Determine the corresponding label file
            label_file_path = img_file.with_suffix('.txt')
                
            # Save image and label in the respective directories
            if contains_bundle:
                image_save_path = self.bundle_images_dir_ / (img_file.stem + ".jpg")
                bundle_count += 1
                with open(self.bundle_labels_dir_ / label_file_path.name, 'w') as f:
                    f.write(label_file_content)
            elif all_single:
                image_save_path = self.single_images_dir_ / (img_file.stem + ".jpg")
                single_count += 1
                with open(self.single_labels_dir_ / label_file_path.name, 'w') as f:
                    f.write(label_file_content)

            # Save the image as .jpg
            cv2.imwrite(str(image_save_path), img)

            # Optionally save the masked image or with bounding boxes
            if self.save_masked_images_:
                if hasattr(results, 'masks') and results.masks:
                    masked_img = self.draw_masks(img, results.masks.data, results.boxes.cls)
                else:
                    masked_img = self.draw_bounding_boxes(img, results.boxes.xyxy, results.boxes.cls)
                masked_img_path = self.masked_images_dir_ / (img_file.stem + ".jpg")
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
    parser.add_argument("--save-masked-images", action="store_true", help="Save images with drawn masks or bounding boxes")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon value for polygon simplification")
    parser.add_argument("--resize-height", type=int, help="Height to resize images before processing")
    parser.add_argument("--resize-width", type=int, help="Width to resize images before processing")

    args = parser.parse_args()

    yolo_inference = YOLOInference(
        model_path=args.model_path,
        image_dir=args.image_dir,
        confidence=0.5,
        save_masked_images=args.save_masked_images,
        epsilon=args.epsilon,
        resize_height=args.resize_height,
        resize_width=args.resize_width
    )

    yolo_inference.run_inference()
