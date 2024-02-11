"""
How to Use the App
---
** pip install Pillow **

* Run the script. The GUI will open.
* Click on "Select Input Directory" to choose the directory with your images.
* Click on "Select Output Directory" to choose the directory where you want to save images.
* Navigate through the images using the left and right arrow keys on your keyboard.
* Press the 's' key to save the currently displayed image to the output directory.
"""
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import os

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.input_dir = None
        self.output_dir = None
        self.images = []
        self.current_image_index = 0
        self.setup_ui()
        self.image_label = None

    def setup_ui(self):
        self.root.title("Image Viewer")

        # Frame for Directory Selection and Instructions
        self.top_frame = Frame(self.root)
        self.top_frame.grid(row=0, column=0, columnspan=2)

        # Input Directory Selection
        self.input_dir_btn = Button(self.top_frame, text="Select Input Directory", command=self.select_input_dir)
        self.input_dir_btn.pack()
        self.input_dir_label = Label(self.top_frame, text="No Input Directory Selected")
        self.input_dir_label.pack()

        # Output Directory Selection
        self.output_dir_btn = Button(self.top_frame, text="Select Output Directory", command=self.select_output_dir)
        self.output_dir_btn.pack()
        self.output_dir_label = Label(self.top_frame, text="No Output Directory Selected")
        self.output_dir_label.pack()

        # Instructions
        self.instructions_label = Label(self.root, text="Instructions:\n- Navigate images with left/right keys\n- Save image with 's' key or button", justify=tk.LEFT)
        self.instructions_label.grid(row=1, column=0, sticky="nsew", padx=10)

        # Image Display Frame
        self.image_frame = Frame(self.root)
        self.image_frame.grid(row=1, column=1, sticky="nsew")

        # Image Display
        self.image_panel = Label(self.image_frame)
        self.image_panel.pack()

        # Image Name Display
        self.image_name_label = Label(self.image_frame, text="")
        self.image_name_label.pack()

        # Save Button
        self.save_button = Button(self.image_frame, text="Save Image", command=self.save_image_button)
        self.save_button.pack()

        # Save Confirmation Label
        self.save_confirmation_label = Label(self.image_frame, text="")
        self.save_confirmation_label.pack()

        # Key Bindings
        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("s", self.save_image_keyboard)

        # Configure grid expansion
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

    def select_input_dir(self):
        self.input_dir = filedialog.askdirectory()
        if self.input_dir:
            self.input_dir_label.config(text=self.input_dir)
            self.load_images()

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            self.output_dir_label.config(text=self.output_dir)

    def load_images(self):
        self.images = [img for img in os.listdir(self.input_dir) if img.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        if self.images:
            self.current_image_index = 0
            self.display_image()

    def display_image(self):
        if not self.images:
            return
        image_path = os.path.join(self.input_dir, self.images[self.current_image_index])
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        self.image_panel.configure(image=img)
        self.image_panel.image = img
        self.image_name_label.config(text=f"Current Image: {self.images[self.current_image_index]}")
        self.save_confirmation_label.config(text="")

    def next_image(self, event):
        if self.images and self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.display_image()

    def previous_image(self, event):
        if self.images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def save_image(self):
        if self.images and self.output_dir:
            current_image_path = os.path.join(self.input_dir, self.images[self.current_image_index])
            output_image_path = os.path.join(self.output_dir, self.images[self.current_image_index])
            img = Image.open(current_image_path)
            img.save(output_image_path)
            self.save_confirmation_label.config(text="Image saved successfully.")

    def save_image_keyboard(self, event):
        self.save_image()

    def save_image_button(self):
        self.save_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
