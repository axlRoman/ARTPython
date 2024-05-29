import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

class ART:
    def __init__(self, vigilance=0.5):
        self.vigilance = vigilance
        self.weights = []

    def train(self, input_vector):
        input_vector = np.array(input_vector)
        
        if not self.weights:
            self.weights.append(input_vector)
            return input_vector.reshape((100, 100))
        
        for i, weight in enumerate(self.weights):
            match = np.sum(np.minimum(input_vector, weight)) / np.sum(input_vector)
            if match >= self.vigilance:
                self.weights[i] = np.minimum(input_vector, weight)
                return self.weights[i].reshape((100, 100))
        
        self.weights.append(input_vector)
        return input_vector.reshape((100, 100))

class ARTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Response Theory (ART) Application")

        self.art = ART()

        self.canvas_size = 400
        self.grid_size = 100

        self.create_widgets()
        self.update_canvas()

    def create_widgets(self):
        self.vigilance_label = tk.Label(self.root, text="Vigilance Parameter:")
        self.vigilance_label.pack()

        self.vigilance_entry = tk.Entry(self.root)
        self.vigilance_entry.pack()
        self.vigilance_entry.insert(0, "0.5")

        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.clear_button = tk.Button(self.root, text="Clear Grid", command=self.clear_grid)
        self.clear_button.pack()

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        self.result_canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.result_canvas.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert("L").resize((self.grid_size, self.grid_size))
            self.image_vector = np.array(self.image).flatten() / 255.0

            try:
                self.art.vigilance = float(self.vigilance_entry.get())
            except ValueError:
                self.art.vigilance = 0.5

            self.result_vector = self.art.train(self.image_vector)
            self.update_canvas()

    def clear_grid(self):
        self.canvas.delete("all")
        self.result_canvas.delete("all")

    def update_canvas(self):
        if hasattr(self, 'image'):
            self.display_image(self.canvas, self.image)

        if hasattr(self, 'result_vector'):
            result_image = Image.fromarray((self.result_vector * 255).astype(np.uint8))
            self.display_image(self.result_canvas, result_image)

    def display_image(self, canvas, image):
        self.tk_image = ImageTk.PhotoImage(image.resize((self.canvas_size, self.canvas_size)))
        canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ARTApp(root)
    root.mainloop()
