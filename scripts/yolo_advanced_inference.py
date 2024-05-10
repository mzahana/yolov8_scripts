import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import networkx as nx
import colorsys
from ultralytics import YOLO

def index_to_color(index, total):
    """Generate a distinct RGB color from the index."""
    # Generate a unique color using HSV tuple and converting it to RGB
    hue = index / total
    saturation = 0.75
    value = 0.95
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

def is_overlapping(box1, box2) -> bool:
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2

    x = x1max >= x2min and x2max >= x1min
    y = y1max >= y2min and y2max >= y1min

    return (x and y)

def is_within_distance(box1, box2, threshold):
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2

    # Calculate the shortest distance between edges of the two boxes
    horizontal_distance = max(0, max(x1min - x2max, x2min - x1max))
    vertical_distance = max(0, max(y1min - y2max, y2min - y1max))

    # Calculate the actual Euclidean distance between the closest edges
    distance = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5

    return distance <= threshold

def check_proximity(boxA, boxB, threshold):
    if is_overlapping(boxA, boxB):
        return True
    return is_within_distance(boxA, boxB, threshold)

def merge_boxes(boxes):
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return (min_x, min_y, max_x, max_y)

class YOLOInference:
    def __init__(self, model_path, input_dir, confidence=0.5, threshold=50, save_original=False, font_size=10):
        self.model_path = model_path
        self.input_dir = Path(input_dir)
        self.confidence = confidence
        self.threshold = threshold
        self.save_original = save_original
        self.font_size = font_size
        
        self.images_dir = self.input_dir.parent / "processed_images"
        self.labels_dir = self.input_dir.parent / "new_labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = YOLO(self.model_path)
        self.model.fuse()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load custom font or default
        try:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)
        except IOError:
            self.font = ImageFont.load_default()

        # Color dictionary to cache class color
        self.color_map = {}
        # Specific color for processed bundles
        self.processed_bundle_color = (255, 105, 180)  # Hot pink

    def get_color(self, class_name, class_idx=None):
        if class_name == 'Processed_Bundle':
            return self.processed_bundle_color
        if class_name not in self.color_map:
            total_classes = len(self.model.names)  # Assuming model.names holds class names
            self.color_map[class_name] = index_to_color(class_idx, total_classes)
        return self.color_map[class_name]

    def cluster_bounding_boxes(self, boxes):
        G = nx.Graph()
        for i, box in enumerate(boxes):
            G.add_node(i, box=box)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if check_proximity(boxes[i], boxes[j], self.threshold):
                    G.add_edge(i, j)
        clusters = list(nx.connected_components(G))
        return clusters
    
    def run_inference(self):
        image_paths = list(self.input_dir.glob('*.png')) + list(self.input_dir.glob('*.jpg'))
        
        for img_path in image_paths:
            results = self.model.predict(source=str(img_path), verbose=False, stream=False, conf=self.confidence, device=self.device)
            results = results[0].cpu()

            if results.boxes:
                boxes = [tuple(box.xyxy[0].tolist()) for box in results.boxes]
                class_names = [self.model.names[int(box.cls)] for box in results.boxes]
                class_indices = [int(box.cls) for box in results.boxes]
                
                clusters = self.cluster_bounding_boxes(boxes)
                merged_boxes = []
                for cluster in clusters:
                    group = [boxes[i] for i in cluster]
                    if len(cluster) > 1:
                        merged_box = merge_boxes(group)
                        merged_boxes.append((merged_box, 'Processed_Bundle'))
                    else:
                        idx = list(cluster)[0]
                        merged_boxes.append((boxes[idx], class_names[idx]))

                original_image = Image.open(img_path)
                processed_image = original_image.copy()
                draw_original = ImageDraw.Draw(original_image)
                draw_processed = ImageDraw.Draw(processed_image)

                for box, class_name, class_idx in zip(boxes, class_names, class_indices):
                    color = self.get_color(class_name, class_idx)
                    draw_original.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
                    draw_original.text((box[0], box[1] - 10), class_name, fill=color, font=self.font)

                for box, class_name in merged_boxes:
                    color = self.get_color(class_name)  # Now fetches specific color for 'Processed_Bundle'
                    draw_processed.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
                    draw_processed.text((box[0], box[1] - 10), class_name, fill=color, font=self.font)

                if self.save_original:
                    total_width = original_image.width * 2
                    new_image = Image.new('RGB', (total_width, original_image.height))
                    new_image.paste(original_image, (0, 0))
                    new_image.paste(processed_image, (original_image.width, 0))
                    new_image.save(self.images_dir / f"{img_path.stem}_combined.jpg")
                else:
                    processed_image.save(self.images_dir / img_path.name)

                with open(self.labels_dir / f"{img_path.stem}.txt", 'w') as file:
                    for box, class_name in merged_boxes:
                        file.write(f"{class_name} {box[0]} {box[1]} {box[2]} {box[3]}\n")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO model inference on a directory of images with optional proximity threshold for bounding box merging.")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file.")
    parser.add_argument("input_dir", type=str, help="Directory containing images for inference.")
    parser.add_argument("--threshold", type=int, default=50, help="Proximity threshold in pixels to consider bounding boxes close. Default is 50 pixels.")
    parser.add_argument("--save_original", action='store_true', help="Save original and processed images side-by-side.")
    parser.add_argument("--font_size", type=int, default=10, help="Font size for class text.")
    args = parser.parse_args()

    yolo_inference = YOLOInference(model_path=args.model_path, input_dir=args.input_dir, threshold=args.threshold, save_original=args.save_original, font_size=args.font_size)
    yolo_inference.run_inference()

if __name__ == "__main__":
    main()