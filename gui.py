import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from art import ART1
from image_processing import load_image, image_to_pattern

class ARTApp:
    def __init__(self, master):
        self.master = master
        master.title("Clasificador de Imágenes ART")
        master.geometry("1280x720")

        self.vigilance = tk.DoubleVar()
        self.vigilance.set(0.5)  # Valor inicial de vigilancia

        self.model = ART1(num_features=10, vigilance=self.vigilance.get())

        # Mapeo de números de clase a nombres de Pokémon
        self.class_to_pokemon = {
            0: "Bulbasaur",
            1: "Charmander",
            2: "Squirtle"
            # Agrega más entradas según sea necesario
        }

        # Frame for modified image (left)
        self.modify_frame = tk.Frame(master, width=426, height=720)
        self.modify_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for controls and results (center)
        self.control_frame = tk.Frame(master, width=426, height=720)
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for classified image (right)
        self.result_frame = tk.Frame(master, width=426, height=720)
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.upload_button = tk.Button(self.control_frame, text="Subir imagen", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.classify_button = tk.Button(self.control_frame, text="Clasificar", command=self.classify_image)
        self.classify_button.pack(pady=10)

        self.reload_button = tk.Button(self.control_frame, text="Re-subir imagen", command=self.reload_image)
        self.reload_button.pack(pady=10)

        self.vigilance_label = tk.Label(self.control_frame, text="Vigilancia")
        self.vigilance_label.pack(pady=10)

        self.vigilance_slider = tk.Scale(self.control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=self.vigilance, command=self.update_vigilance)
        self.vigilance_slider.pack(pady=10)

        self.result_label = tk.Label(self.control_frame, text="")
        self.result_label.pack(pady=10)

        self.modify_image_label = tk.Label(self.modify_frame)
        self.modify_image_label.pack(pady=10)

        self.classified_image_label = tk.Label(self.result_frame)
        self.classified_image_label.pack(pady=10)

        # Add mouse click and drag events to the modified image
        self.modify_image_label.bind("<Button-1>", self.modify_pixel)
        self.modify_image_label.bind("<B1-Motion>", self.modify_pixel)

        self.exit_button = tk.Button(self.control_frame, text="Salir", command=self.quit_program)
        self.exit_button.pack(side=tk.BOTTOM, anchor=tk.SE, pady=10)

        self.center_widgets(self.control_frame)
        self.center_widgets(self.modify_frame)
        self.center_widgets(self.result_frame)

    def center_widgets(self, frame):
        for widget in frame.winfo_children():
            widget.pack_configure(anchor=tk.CENTER)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = load_image(file_path)
            self.pattern = image_to_pattern(self.image)
            self.show_image(self.image, self.modify_image_label)
            self.modified_image = self.image.copy()
            self.show_image(self.modified_image, self.modify_image_label)

    def reload_image(self):
        if hasattr(self, 'modified_image'):
            self.image = self.modified_image.copy()
            self.pattern = image_to_pattern(self.image)
            self.show_image(self.image, self.modify_image_label)
            messagebox.showinfo("Cargar la imagen nuevamente", "Se modifico la imagen exitosamente")
        else:
            messagebox.showerror("Error", "No modified image available to reload.")

    def show_image(self, image, label):
        image_pil = Image.fromarray(image)
        scaled_image_pil = image_pil.resize((426, 426), Image.NEAREST)  # Escalar la imagen para visualización
        image_tk = ImageTk.PhotoImage(scaled_image_pil)
        label.config(image=image_tk)
        label.image = image_tk  # Keep a reference to avoid garbage collection

    def classify_image(self):
        if hasattr(self, 'pattern'):
            classification = self.model.predict(self.pattern)
            if classification == -1:
                self.model.train([self.pattern])
                classification = len(self.model.weights) - 1
            pokemon_name = self.class_to_pokemon.get(classification, "Desconocido")
            self.result_label.config(text=f"Imagen clasificada como {pokemon_name}")
            self.show_classified_image(classification)
        else:
            messagebox.showerror("Error", "Ninguna imagen se subió")

    def show_classified_image(self, classification):
        # Load the example image from the corresponding class directory
        class_name = self.class_to_pokemon.get(classification, "Desconocido")
        class_dir = os.path.join("clases", class_name)
        if os.path.isdir(class_dir):
            example_image_path = os.path.join(class_dir, os.listdir(class_dir)[0])  # Take the first image as an example
            example_image = load_image(example_image_path)
            self.show_image(example_image, self.classified_image_label)

    def modify_pixel(self, event):
        if hasattr(self, 'modified_image'):
            x = event.x * 100 // self.modify_image_label.winfo_width()
            y = event.y * 100 // self.modify_image_label.winfo_height()
            if 0 <= x < 100 and 0 <= y < 100:  # Ensure x and y are within bounds
                self.modified_image[y, x] = 255 if self.modified_image[y, x] == 0 else 0
                self.show_image(self.modified_image, self.modify_image_label)

    def update_vigilance(self, val):
        self.model.vigilance = self.vigilance.get()

    def quit_program(self):
        self.master.quit()
        self.master.destroy()

root = tk.Tk()
app = ARTApp(root)
root.mainloop()
