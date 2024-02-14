"""
YOLO Model Inference and Image/Label Sorting Script

This script is designed to automate the process of running object detection on a dataset of images using a pre-trained YOLO model, and then categorizing these images (and their corresponding label files) into different directories based on the presence of user-specified object classes. The script operates on a structured dataset located within a specified parent directory, which must contain 'images' and 'labels' subdirectories for input data.

Key Features:
- Utilizes a pre-trained YOLO model to perform object detection on each image in the specified dataset.
- Filters and categorizes images into 'bundle_images' or 'single_images' based on whether detected objects match a set of desired classes defined by the user. Corresponding label files are sorted similarly into 'bundle_labels' or 'single_labels'.
- Automatically creates necessary output subdirectories within the parent directory for organized storage of filtered images and labels.
- Provides flexibility in specifying the inference confidence threshold and desired object classes to tailor the sorting process to specific needs.

Usage:
To use this script, you need to provide the path to the YOLO model file, the path to the parent directory containing your 'images' and 'labels' subdirectories, an optional confidence threshold for detections, and a list of object classes of interest. The script then processes each image, performs inference, and sorts the images and labels into the appropriate directories based on the detection results.

Example:
desired_classes = ['person', 'bicycle', 'car']
yolo_inference = YOLOInference(
    model_path='/path/to/yolo/model.pt',
    parent_dir='/path/to/dataset/parent/directory',
    confidence=0.5,  # Optional confidence threshold
    desired_classes=desired_classes
)
yolo_inference.run_inference()

Dependencies:
- ultralytics YOLO package for object detection.
- PyTorch for deep learning operations.
- Standard Python libraries: pathlib, shutil, os for file and directory operations.

Ensure the ultralytics YOLO package, PyTorch, and necessary file handling libraries are installed and correctly set up in your Python environment before executing this script. The script is designed to work with datasets where each image in the 'images' subdirectory has a corresponding label file with the same name (but with a .txt extension) in the 'labels' subdirectory.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

class YOLOInference:
    def __init__(self, model_path, parent_dir, confidence=0.5, desired_classes=None):
        self.model_path_ = model_path
        self.parent_dir_ = Path(parent_dir)
        self.images_dir_ = self.parent_dir_ / "images"
        self.labels_dir_ = self.parent_dir_ / "labels"
        self.confidence_ = confidence
        self.desired_classes_ = desired_classes or []
        
        # Define output directories for images and labels based on classification
        self.bundle_images_dir_ = self.parent_dir_ / "bundle_images"
        self.bundle_labels_dir_ = self.parent_dir_ / "bundle_labels"
        self.single_images_dir_ = self.parent_dir_ / "single_images"
        self.single_labels_dir_ = self.parent_dir_ / "single_labels"
        
        # Ensure all output directories exist
        for dir in [self.bundle_images_dir_, self.bundle_labels_dir_, self.single_images_dir_, self.single_labels_dir_]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model_ = YOLO(self.model_path_)
        self.model_.fuse()
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def run_inference(self):
        total_images = 0

        # Iterate over images in the images directory
        image_paths = list(self.images_dir_.glob('*.png')) + list(self.images_dir_.glob('*.jpg'))
        total_images = len(image_paths)
        
        for img_file in image_paths:
            # Perform inference on each image
            results = self.model_.predict(
                source=str(img_file),
                verbose=False,
                stream=False,
                conf=self.confidence_,
                device=self.device_
            )
            results = results[0].cpu()
            contains_desired_class = False
            
            if results.boxes:
                for box_data in results.boxes:
                    class_name = self.model_.names[int(box_data.cls)]
                    if class_name in self.desired_classes_:
                        contains_desired_class = True
                        break  # Found a desired class, no need to check further
                
                # Determine the corresponding label file
                label_file = self.labels_dir_ / img_file.with_suffix('.txt').name
                
                if contains_desired_class:
                    # Save image and label in the bundle directories
                    shutil.copy(img_file, self.bundle_images_dir_)
                    if label_file.exists():
                        shutil.copy(label_file, self.bundle_labels_dir_)
                else:
                    # Save image and label in the single directories
                    shutil.copy(img_file, self.single_images_dir_)
                    if label_file.exists():
                        shutil.copy(label_file, self.single_labels_dir_)

        # Print the summary
        print(f"Total images processed: {total_images}")

# Example usage
desired_classes = ['Bundle', 'Top-Most-Item']
yolo_inference = YOLOInference(
    model_path='/home/mzahana/datasets/Silki/Bundle_Detection.v3i.yolov8/runs/segment/yolov8n_v3i_ep109/weights/best.pt',
    parent_dir='/home/mzahana/datasets/Silki/Bundle_Detection.v3i.yolov8/merged_data/train',
    confidence=0.5,
    desired_classes=desired_classes
)

yolo_inference.run_inference()
