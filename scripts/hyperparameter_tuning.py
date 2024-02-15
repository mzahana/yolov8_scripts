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