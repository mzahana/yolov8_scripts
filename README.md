# yolov8_scripts
Collection of scripts that are useful for YOLOv8 projects.

# Installation
To install YOLOv8, you can use the following commands.
```bash
sudo apt install -y python3-pip
pip install ultralytics

```


# Segmentation

## Training

### Train a new model from YAML and start training from scratch
```bash
yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640 batch=16

# Start training from a pretrained *.pt model
yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=16

# Build a new model from YAML, transfer pretrained weights to it and start training
yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640 batch=16
```

* To resume training
```bash
# Resume an interrupted training
yolo train resume model=path/to/last.pt
```

## Predicting
```bash
yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model

yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
```

# Example commands
```bash
yolo segment train data=/home/mzahana/datasets/Silki/Bundle_Detection.v1i.yolov8/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=128
```

# Scripts

## `yolo_inference.py`

### Description

The YOLO Inference Script automates object detection and filtering on a collection of images using a pre-trained YOLOv8 model. It processes images from a specified input directory, applies the YOLO model to detect objects within these images, and filters out those containing specific predefined classes of interest. This script is designed for projects requiring automated image categorization or filtering based on object detection. It supports `.png` and `.jpg` formats and offers customization options for the model path, confidence threshold, and classes of interest.

### Features

- **Model Flexibility**: Compatible with any YOLOv8 model.
- **Selective Saving**: Saves only images containing specified object classes.
- **Customizable Confidence Threshold**: Allows setting a confidence level for object detection.
- **Automatic Output Directory**: Automatically creates an output directory if none is specified.
- **GPU Acceleration**: Leverages GPU for enhanced processing speed, if available.

### Installation Requirements

Ensure Python 3.x is installed along with the necessary packages:
- `torch`
- `ultralytics`

These can be installed using pip:

```sh
pip install torch ultralytics
```

### Script Configuration
Configure the script by setting the following parameters according to your project needs:

* `model_path`: Path to the YOLOv8 model file (.pt).
* `input_dir`: Directory containing images for processing.
* `output_dir`: (Optional) Directory for saving filtered images.
* `confidence`: (Optional) Confidence threshold for detection.
* `desired_classes`: List of object classes to filter images by.

### Running the Script
Execute the script with your configuration:

```python
from YOLOInferenceScript import YOLOInference

yolo_inference = YOLOInference(
    model_path='/path/to/your/model.pt',
    input_dir='/path/to/your/input/directory',
    output_dir='/path/to/your/output/directory',  # Optional
    confidence=0.5,
    desired_classes=['class1', 'class2']
)

yolo_inference.run_inference()
```

### Output
The script will:

* Save the filtered images to the specified output directory.
* Print the total count of processed images, the number saved, and the percentage saved.

# References
* Segmentation: https://docs.ultralytics.com/tasks/segment/
* Training tips: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/
* Performance metrics: https://docs.ultralytics.com/guides/yolo-performance-metrics/
* Prediction: https://docs.ultralytics.com/modes/predict/
