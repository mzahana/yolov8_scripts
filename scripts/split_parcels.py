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
