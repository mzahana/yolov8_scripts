"""
YOLO Inference Script

This script is designed to perform object detection and filtering on a collection of images using a pre-trained YOLOv8 model. It processes images from a specified input directory, applying the YOLO model to detect objects within these images. Only images containing specific object classes of interest, defined by the user, are saved to an output directory.

Key Features:
- Loads a YOLOv8 model from a specified .pt file path.
- Processes all .png and .jpg images within a specified input directory.
- Filters and saves images that contain any of the predefined object classes of interest to an output directory. The output directory is either specified by the user or automatically created in the parent directory of the input folder.
- Counts the total number of images processed, the number of images saved, and calculates the percentage of images saved based on the criteria.

Usage:
To use this script, define the path to the YOLOv8 model, the input directory containing the images to be processed, the (optional) output directory where filtered images should be saved, the inference confidence threshold, and a list of desired object classes. The script then automatically handles the detection, filtering, and saving of images.

Example:
desired_classes = ['class1', 'class2', 'class3']
yolo_inference = YOLOInference(
    model_path='/path/to/model.pt',
    input_dir='/path/to/input/directory',
    output_dir='/path/to/output/directory',  # Optional
    confidence=0.5,
    desired_classes=desired_classes
)
yolo_inference.run_inference()

Dependencies:
- ultralytics YOLO package
- PyTorch
- pathlib, shutil, os for directory and file handling

Ensure that the ultralytics YOLO package and PyTorch are installed and properly configured before running this script.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

class YOLOInference:
    def __init__(self, model_path, input_dir, output_dir=None, confidence=0.5, desired_classes=None):
        self.model_path_ = model_path
        self.input_dir_ = input_dir
        self.confidence_ = confidence
        self.desired_classes_ = desired_classes or []
        
        if output_dir is None:
            self.output_dir_ = Path(input_dir).parent / "output"
        else:
            self.output_dir_ = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model_ = YOLO(self.model_path_)
        self.model_.fuse()
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def run_inference(self):
        total_images = 0
        saved_images = 0

        # Iterate over images in the input directory for both .png and .jpg files
        image_paths = list(Path(self.input_dir_).glob('*.png')) + list(Path(self.input_dir_).glob('*.jpg'))
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
            # Check if any detected object's class name is in the desired set
            if results.boxes:
                for box_data in results.boxes:
                    class_name = self.model_.names[int(box_data.cls)]
                    if class_name in self.desired_classes_:
                        # Save image if it contains objects of the desired classes
                        shutil.copy(img_file, self.output_dir_)
                        saved_images += 1
                        break  # Stop checking other boxes for this image as it's already saved

        # Print the summary
        print(f"Total images in the input folder: {total_images}")
        print(f"Number of images saved: {saved_images}")
        if total_images > 0:
            percentage_saved = (saved_images / total_images) * 100
            print(f"Percentage of images saved: {percentage_saved:.2f}%")
        else:
            print("No images found in the input folder.")

# Example usage
desired_classes = ['Adjacent-Bundle', 'Complex-Bundle', 'Damaged-Item', 'Sequential-Bundle', 'Stacked-Bundle', 'Top-Most-Item']
yolo_inference = YOLOInference(
    model_path='/home/mzahana/datasets/Silki/Bundle_Detection.v3i.yolov8/runs/segment/yolov8n_v3i_ep109/weights/best.pt',
    input_dir='/home/mzahana/datasets/Silki/raw_data/20240129',
    output_dir='/home/mzahana/datasets/Silki/raw_data/20240129_filtered_2',
    confidence=0.5,
    desired_classes=desired_classes
)

yolo_inference.run_inference()
