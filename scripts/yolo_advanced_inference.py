import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import networkx as nx
import colorsys
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from ultralytics import YOLO

def index_to_color(index, total):
    """Generate a distinct RGB color from the index."""
    # Generate a unique color using HSV tuple and converting it to RGB
    hue = index / total
    saturation = 0.75
    value = 0.95
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

def bbox_to_polygon(bbox):
    """Convert a bounding box (x1, y1, x2, y2) to a polygon."""
    x1, y1, x2, y2 = bbox
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def convert_mask_to_polygon_points(mask):
    """
    Convert a mask from YOLOv8 to a list of (x, y) tuples for each point in the mask.

    Parameters:
        mask: The mask object from YOLOv8 containing xy points.

    Returns:
        List of tuples representing the polygon's vertices.
    """
    return [(float(point[0]), float(point[1])) for point in mask.xy[0].tolist()]

def are_polygons_in_proximity(mask1, mask2, threshold):
    """
    Check if two masks defined by polygons are in proximity based on a threshold distance.

    Parameters:
        mask1 (list of tuples): Points [(x1, y1), (x2, y2), ..., (xn, yn)] defining the first polygon.
        mask2 (list of tuples): Points [(x1, y1), (x2, y2), ..., (xn, yn)] defining the second polygon.
        threshold (float): Maximum distance between polygons to consider them as proximate.

    Returns:
        bool: True if polygons overlap or are within the threshold distance, False otherwise.
    """
    poly1 = Polygon(mask1)
    poly2 = Polygon(mask2)

    # Check for overlap
    if poly1.intersects(poly2):
        return True

    # Compute the minimum distance between the two polygons
    min_distance = poly1.distance(poly2)

    return min_distance <= threshold

def are_bbx_in_proximity(box1, box2, threshold) -> bool:
    box1_polygon = bbox_to_polygon(box1)
    # print(box1_polygon)
    box2_polygon = bbox_to_polygon(box2)
    # print(box2_polygon)

    return are_polygons_in_proximity(box1_polygon, box2_polygon, threshold)

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

def min_horizontal_distance(box1, box2):
    """
    Calculate the minimum horizontal distance between two bounding boxes.

    Parameters:
        box1 (tuple): Coordinates (x1min, y1min, x1max, y1max) for the first box.
        box2 (tuple): Coordinates (x2min, y2min, x2max, y2max) for the second box.

    Returns:
        float: The minimum horizontal distance between the two boxes.
    """
    x1min, _, x1max, _ = box1
    x2min, _, x2max, _ = box2
    if x1max < x2min:
        return x2min - x1max  # Distance from box1 to box2
    elif x2max < x1min:
        return x1min - x2max  # Distance from box2 to box1
    else:
        return 0  # Overlapping or touching

def are_bbx_in_horizontal_proximity(box1, box2, threshold):
    """
    Check if two bounding boxes are in horizontal proximity based on a threshold.

    Parameters:
        box1 (tuple): First bounding box.
        box2 (tuple): Second bounding box.
        threshold (float): Threshold for deciding proximity.

    Returns:
        bool: True if the horizontal distance is less than or equal to the threshold.
    """
    return min_horizontal_distance(box1, box2) <= threshold

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

    # def cluster_bounding_boxes(self, boxes):
    #     G = nx.Graph()
    #     for i, box in enumerate(boxes):
    #         G.add_node(i, box=box)
    #     for i in range(len(boxes)):
    #         for j in range(i + 1, len(boxes)):
    #             # if check_proximity(boxes[i], boxes[j], self.threshold):
    #             if are_bbx_in_proximity(boxes[i], boxes[j], self.threshold):
    #                 G.add_edge(i, j)
    #     clusters = list(nx.connected_components(G))
    #     return clusters
    def cluster_bounding_boxes(self, boxes):
        """
        Group bounding boxes based on horizontal proximity.

        Parameters:
            boxes (list of tuples): List of bounding boxes.
            threshold (float): Threshold to consider boxes as close.

        Returns:
            list of sets: Each set contains indices of boxes in the same cluster.
        """
        G = nx.Graph()
        for i, box in enumerate(boxes):
            G.add_node(i, box=box)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if are_bbx_in_horizontal_proximity(boxes[i], boxes[j], self.threshold):
                    G.add_edge(i, j)
        return list(nx.connected_components(G))
    
    def cluster_masks(self, masks):
        G = nx.Graph()
        for i, mask in enumerate(masks):
            G.add_node(i, mask=mask)
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                # if check_proximity(boxes[i], boxes[j], self.threshold):
                polygon1= convert_mask_to_polygon_points(masks[i])
                polygon2 = convert_mask_to_polygon_points(masks[j])
                if are_polygons_in_proximity(polygon1, polygon2, self.threshold):
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
                
                # clusters = self.cluster_bounding_boxes(boxes)
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