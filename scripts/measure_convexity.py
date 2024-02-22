import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from time import time

def plot_bounding_boxes(image, bounding_boxes):
    """
    Draws bounding boxes on the image.

    Parameters:
    - image: The original image on which to draw the bounding boxes.
    - bounding_boxes: A list of bounding box coordinates, where each bounding box is defined as [x1, y1, x2, y2].

    Returns:
    - The image with bounding boxes drawn on it.
    """
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()

    # Loop through all bounding boxes and draw them on the image
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle with green color and 2px thickness

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.axis('off')
    plt.show()

def plot_bounding_box(image, bounding_box):
    """
    Draws bounding boxes on the image.

    Parameters:
    - image: The original image on which to draw the bounding boxes.
    - bounding_boxes: A list of bounding box coordinates, where each bounding box is defined as [x1, y1, x2, y2].

    Returns:
    - The image with bounding boxes drawn on it.
    """
    # Create a copy of the image to draw on
    im = image.copy()

    # Loop through all bounding boxes and draw them on the image
    
    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle with green color and 2px thickness

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.axis('off')
    plt.show()

def calculate_convexity_and_plot(image, polygon_points):
    # Convert points to a suitable format
    points = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Calculate the area of the original polygon
    area_polygon = cv2.contourArea(points)
    
    # Compute the convex hull of the polygon
    hull_points = cv2.convexHull(points)
    
    # Calculate the area of the convex hull
    area_hull = cv2.contourArea(hull_points)
    
    # Calculate the convexity (ratio of the areas)
    convexity = area_polygon / area_hull if area_hull > 0 else 0
    
    # Create a copy of the image to draw on
    image_copy = image.copy()
    
    # Draw the original polygon
    cv2.polylines(image_copy, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Draw the convex hull
    cv2.polylines(image_copy, [hull_points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Display the convexity value on the image
    cv2.putText(image_copy, f"Convexity: {convexity:.2f}", tuple(hull_points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Plot the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def calculate_convexity_and_plot_for_all_masks(image, mask_list):
    # Create a copy of the image to draw on
    t1 = time()
    image_copy = image.copy()
    
    for polygon_points in mask_list:
        # Convert points to a suitable format
        points = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Calculate the area of the original polygon
        area_polygon = cv2.contourArea(points)
        
        # Compute the convex hull of the polygon
        hull_points = cv2.convexHull(points)
        
        # Calculate the area of the convex hull
        area_hull = cv2.contourArea(hull_points)
        
        # Calculate the convexity (ratio of the areas)
        convexity = area_polygon / area_hull if area_hull > 0 else 0
        
        # Draw the original polygon
        cv2.polylines(image_copy, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Draw the convex hull
        cv2.polylines(image_copy, [hull_points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Display the convexity value on the image
        cv2.putText(image_copy, f"Convexity: {convexity:.2f}", tuple(hull_points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    print(f"time to compute convecity = {(time() - t1)*1000} ms")
    # Plot the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

model_path='/home/mzahana/datasets/bundle_detection/v7_filtered-allclasses-withnightimages/runs/segment/v8x.seg.v7.filtered.allclasses.withnightimages/weights/best.pt'
# Load a model
yolo_model = YOLO(model_path)  # pretrained model

sam_model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

# Run inference on an image
img_path='/home/mzahana/src/sam/bundle_1.jpg'
img = cv2.imread(img_path)
yolo_results = yolo_model(img, conf=0.6, device=0)  # return a list of Results objects
yolo_results = yolo_results[0].cpu()
yolo_results.show()
print("number of boxes = ", len(yolo_results.boxes))


# Process results list
# mask_list=[]
# if results.masks:
#     for mask in results.masks:
#         # print(type(mask))
#         mask_points = [ (xy[0], xy[1])for xy in mask.xy[0].tolist()]
#         calculate_convexity_and_plot(img, mask_points)
#         mask_list.append(mask_points)

# calculate_convexity_and_plot_for_all_masks(img, mask_list)

if yolo_results.boxes:
    for box in yolo_results.boxes:
        bx = box.xyxy[0].tolist()
        bx=[int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])]

        # Run FastSAM inference on an image
        everything_results = sam_model(img, device=0, retina_masks=True, imgsz=640, conf=0.7, iou=0.2, save=False, verbose=False)
        prompt_process = FastSAMPrompt(img, everything_results, device=0)
        results = prompt_process.box_prompt(bbox=bx)
        results = results[0].cpu()
        for mask in results.masks:
            mask_points = [ (xy[0], xy[1])for xy in mask.xy[0].tolist()]
            calculate_convexity_and_plot(img, mask_points)

        plot_bounding_box(img, bx)


# Example usage:
# polygon_points = [(50, 50), (50, 150), (150, 150), (150, 50)] # Example for a square (convex shape)
# convexity = calculate_convexity(polygon_points)
# print(f"Convexity for a convex shape: {convexity}")

# polygon_points = [(50, 50), (50, 150), (125, 125), (150, 150), (150, 50)] # Example for a non-convex shape
# convexity = calculate_convexity(polygon_points)
# print(f"Convexity for a non-convex shape: {convexity}")
