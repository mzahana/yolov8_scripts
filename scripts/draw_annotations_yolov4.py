"""
This script processes images from specific subdirectories ('train', 'test', 'valid') within a given dataset directory.
Each subdirectory should contain an '_annotations.txt' file with annotations for the images and a '_classes.txt' file 
with class labels. The script reads these files to draw bounding boxes and class names on the images based on the 
annotations provided. It saves the annotated images in new subdirectories ('annotated_train', 'annotated_test', 
'annotated_valid') within the base directory.

Usage:
- Ensure each dataset subdirectory ('train', 'test', 'valid') has '_annotations.txt' and '_classes.txt'.
- The '_annotations.txt' should contain lines with an image filename followed by space-separated bounding box data.
  Each bounding box is formatted as 'x1,y1,x2,y2,class_id'.
- The '_classes.txt' should list class names, each on a new line, corresponding to the class IDs used in the annotations.
- Images are expected to be located in the same directory as their corresponding '_annotations.txt'.

To run the script:
1. Install Python and OpenCV. You can install OpenCV using `pip install opencv-python`.
2. Place this script in a directory.
3. Run the script via command line: `python annotate_images.py /path/to/your/dataset`
   Replace `/path/to/your/dataset` with the actual path to your dataset directory.
"""

import cv2
import os

def draw_annotations(base_path):
    # Define the data set folders and their corresponding new folders for annotated images
    data_sets = ['train', 'test', 'valid']
    new_folders = ['annotated_train', 'annotated_test', 'annotated_valid']

    for set_name, new_folder in zip(data_sets, new_folders):
        dir_path = os.path.join(base_path, set_name)
        new_dir_path = os.path.join(base_path, new_folder)
        image_count = 0

        # Create directory for annotated images if it does not exist
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)

        annotations_path = os.path.join(dir_path, '_annotations.txt')
        classes_path = os.path.join(dir_path, '_classes.txt')

        # Load class labels
        with open(classes_path, 'r') as file:
            classes = [line.strip() for line in file.readlines()]

        # Read annotations
        with open(annotations_path, 'r') as file:
            annotations = file.readlines()

        print(f"Processing images in {set_name}...")

        for annotation in annotations:
            parts = annotation.strip().split(' ')
            image_name = parts[0]
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue

            # Draw each bounding box
            for box in parts[1:]:
                x1, y1, x2, y2, class_id = map(int, box.split(','))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_label = classes[class_id]
                cv2.putText(image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the annotated image in the new directory
            cv2.imwrite(os.path.join(new_dir_path, image_name), image)
            image_count += 1

        print(f"Done! Processed {image_count} images in {set_name}.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    draw_annotations(base_directory)

