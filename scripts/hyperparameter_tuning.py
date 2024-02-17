"""References
1. https://docs.ultralytics.com/guides/hyperparameter-tuning/
2. https://docs.ultralytics.com/integrations/ray-tune/#custom-search-space-example

# Install and update Ultralytics and Ray Tune packages
pip install -U ultralytics "ray[tune]"

# Optionally install W&B for logging
pip install wandb

Example 1 with use_ray=True:
from ultralytics import YOLO

# Load a YOLOv8n model
model = YOLO('yolov8n.pt')

# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(data='coco8.yaml', use_ray=True)


Example 2: Custom serach space
--
from ultralytics import YOLO

# Define a YOLO model
model = YOLO("yolov8n.pt")

# Run Ray Tune on the model
result_grid = model.tune(data="coco128.yaml",
                         space={"lr0": tune.uniform(1e-5, 1e-1)},
                         epochs=50,
                         use_ray=True)
"""

from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='coco8.yaml', epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)

#Example: Training using tuned hyperparameters
# yolo segment train data=/home/mzahana/datasets/Silki/bundle_detection/Bundle_Detection.v6-allclasses-norotation.yolov8/data.yaml model=/home/mzahana/datasets/Silki/bundle_detection/Bundle_Detection.v6-allclasses-norotation.yolov8/runs/segment/tuned/tune/weights/best.pt  optimizer=AdamW epochs=100 imgsz=640 batch=24 optimizer=AdamW  lr0=0.00249 lrf=0.0098 momentum=0.71166 weight_decay=0.00023 warmup_epochs=3.24729 warmup_momentum=0.44252 box=3.32974 cls=0.5337 dfl=1.40185 hsv_h=0.01715 hsv_s=0.54261 hsv_v=0.25993 degrees=0.0 translate=0.0636 scale=0.12643 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.439 mosaic=0.90559 mixup=0.0 copy_paste=0.0