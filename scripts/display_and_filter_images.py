"""
Install
pip install Pillow tk

This script implements a simple GUI to display detections results as iamges and the corresponding raw image side by side.
It is assumed that detection resulta images are in a single directory, and the corresponding raw images  in a different directory.
The user can navifate the detection iamges using arrows keys. If an image needs to be saved, the user can use the 's' key or   hit the save button.
It will be saved in the selected save directory.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Tool")
        self.root.geometry("1200x800")

        self.detections_dir = ""
        self.raw_images_dir = ""
        self.saved_images_dir = ""

        self.detections_images = []
        self.current_image_index = 0

        # Setup UI components
        self.setup_ui()

        # Bind keyboard events
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("s", self.save_image)

    def setup_ui(self):
        # Buttons for directory selection
        tk.Button(self.root, text="Select Detections Directory", command=self.select_detections_dir).pack(pady=5)
        self.detections_dir_label = tk.Label(self.root, text="No directory selected")
        self.detections_dir_label.pack()

        tk.Button(self.root, text="Select Raw Images Directory", command=self.select_raw_images_dir).pack(pady=5)
        self.raw_images_dir_label = tk.Label(self.root, text="No directory selected")
        self.raw_images_dir_label.pack()

        tk.Button(self.root, text="Select Saved Images Directory", command=self.select_saved_images_dir).pack(pady=5)
        self.saved_images_dir_label = tk.Label(self.root, text="No directory selected")
        self.saved_images_dir_label.pack()

        # Image display areas
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.left_image_label = tk.Label(self.image_frame)
        self.left_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_image_label = tk.Label(self.image_frame)
        self.right_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Current image paths
        self.left_image_path_label = tk.Label(self.root, text="Detection Image Path")
        self.left_image_path_label.pack()
        self.right_image_path_label = tk.Label(self.root, text="Raw Image Path")
        self.right_image_path_label.pack()

        # Feedback label
        self.feedback_label = tk.Label(self.root, text="Feedback")
        self.feedback_label.pack()

        # Save button
        tk.Button(self.root, text="Save Image", command=self.save_image).pack(pady=5)

    def select_detections_dir(self):
        self.detections_dir = filedialog.askdirectory()
        if self.detections_dir:
            self.detections_dir_label.config(text=self.detections_dir)
            self.load_images()

    def select_raw_images_dir(self):
        self.raw_images_dir = filedialog.askdirectory()
        if self.raw_images_dir:
            self.raw_images_dir_label.config(text=self.raw_images_dir)

    def select_saved_images_dir(self):
        self.saved_images_dir = filedialog.askdirectory()
        if self.saved_images_dir:
            self.saved_images_dir_label.config(text=self.saved_images_dir)

    def load_images(self):
        self.detections_images = [img for img in os.listdir(self.detections_dir) if img.endswith(('.png', '.jpg'))]
        if self.detections_images:
            self.display_image(0)

    def display_image(self, index):
        if 0 <= index < len(self.detections_images):
            self.current_image_index = index
            detection_image_path = os.path.join(self.detections_dir, self.detections_images[index])
            raw_image_path = os.path.join(self.raw_images_dir, self.detections_images[index])

            try:
                detection_image = Image.open(detection_image_path)
                raw_image = Image.open(raw_image_path)
            except FileNotFoundError:
                self.feedback_label.config(text="Matching raw image not found, skipping...")
                return

            detection_photo = ImageTk.PhotoImage(detection_image)
            raw_photo = ImageTk.PhotoImage(raw_image)

            self.left_image_label.config(image=detection_photo)
            self.left_image_label.image = detection_photo
            self.left_image_path_label.config(text=detection_image_path)

            self.right_image_label.config(image=raw_photo)
            self.right_image_label.image = raw_photo
            self.right_image_path_label.config(text=raw_image_path)
        else:
            if index < 0:
                self.feedback_label.config(text="Reached the beginning.")
            else:
                self.feedback_label.config(text="Reached the end.")

    def next_image(self, event=None):
        self.display_image(self.current_image_index + 1)

    def prev_image(self, event=None):
        self.display_image(self.current_image_index - 1)

    def save_image(self, event=None):
        if self.saved_images_dir and self.raw_images_dir and self.detections_images:
            current_image_name = self.detections_images[self.current_image_index]
            source_path = os.path.join(self.raw_images_dir, current_image_name)
            destination_path = os.path.join(self.saved_images_dir, current_image_name)

            try:
                Image.open(source_path).save(destination_path)
                self.feedback_label.config(text=f"Saved {current_image_name}")
            except FileNotFoundError:
                self.feedback_label.config(text="Image not found, cannot save.")
        else:
            self.feedback_label.config(text="Please select all directories.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()
