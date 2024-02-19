import os
import shutil

def filter_dataset(parent_dir, output_dir, class_ids):
    # Create output directories if they don't exist
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Counters for total and filtered images
    total_images = 0
    filtered_images = 0

    # Iterate through train, valid, and test directories
    for dataset in ['train', 'valid', 'test']:
        dataset_dir = os.path.join(parent_dir, dataset)
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as file:
                lines = file.readlines()
                # Increment total images count
                total_images += 1
                # Check if any object's class ID is in the user-defined set
                if any(int(line.split()[0]) in class_ids for line in lines):
                    # Increment filtered images count
                    filtered_images += 1
                    # Copy corresponding image and label file to output directory
                    image_file = label_file.replace('.txt', '.jpg')  # Assuming images are in .jpg format
                    shutil.copy(os.path.join(images_dir, image_file), output_images_dir)
                    shutil.copy(label_path, output_labels_dir)

    # Print out the total and filtered counts
    print(f'Total images processed: {total_images}')
    print(f'Total filtered images saved: {filtered_images}')
    print(f'Percentage of filtered images {filtered_images/total_images*100} %')

# Example usage
parent_dir = '/home/mzahana/datasets/Silki/bundle_detection/Bundle_Detection.v7-allclasses-withnightimages.yolov8'  # Replace with your dataset path
output_dir = '/home/mzahana/datasets/Silki/bundle_detection/v7_filtered'  # Replace with your desired output path
class_ids = {0, 1, 2, 4, 5}  # Define your set of class IDs here

filter_dataset(parent_dir, output_dir, class_ids)
