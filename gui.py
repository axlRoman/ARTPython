import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from art import ART1
from image_processing import load_image, image_to_pattern

class ARTApp:
    def __init__(self, master):
        self.master = master
        master.title("ART Image Classifier")

        self.model = ART1(num_features=10000, vigilance=0.8)

        # Frame for image
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(side=tk.LEFT)

        # Frame for buttons and results
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.upload_button = tk.Button(self.control_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.classify_button = tk.Button(self.control_frame, text="Classify Image", command=self.classify_image)
        self.classify_button.pack()

        self.result_label = tk.Label(self.control_frame, text="")
        self.result_label.pack()

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = load_image(file_path)
            self.pattern = image_to_pattern(self.image)
            self.show_image(self.image)

    def show_image(self, image):
        # Convert the numpy array to an Image object and then to PhotoImage
        image_pil = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk  # Keep a reference to avoid garbage collection

    def classify_image(self):
        if hasattr(self, 'pattern'):
            classification = self.model.predict(self.pattern)
            if classification == -1:
                self.model.train([self.pattern])
                classification = len(self.model.weights) - 1
            self.result_label.config(text=f"Image classified as class {classification}")
        else:
            messagebox.showerror("Error", "No image uploaded")
