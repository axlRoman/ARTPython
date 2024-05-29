import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from art import ART1
from image_processing import load_image, image_to_pattern

class ARTApp:
    def __init__(self, master):
        self.master = master
        master.title("Clasificador de Imagenes ART")

        self.model = ART1(num_features=10000, vigilance=0.5)

        # Mapeo de números de clase a nombres de Pokémon
        self.class_to_pokemon = {
            0: "Bulbasaur",
            1: "Charmander",
            2: "Squirtle"
            # Agrega más entradas según sea necesario
        }

        # Frame for original image
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(side=tk.LEFT)

        # Frame for controls and results
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for modified image
        self.modify_frame = tk.Frame(master)
        self.modify_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.upload_button = tk.Button(self.control_frame, text="Subir imagen", command=self.upload_image)
        self.upload_button.pack()

        self.classify_button = tk.Button(self.control_frame, text="Clasificar", command=self.classify_image)
        self.classify_button.pack()

        self.reload_button = tk.Button(self.control_frame, text="Re-subir imagen", command=self.reload_image)
        self.reload_button.pack()

        self.result_label = tk.Label(self.control_frame, text="")
        self.result_label.pack()

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.modify_image_label = tk.Label(self.modify_frame)
        self.modify_image_label.pack()

        # Add mouse click and drag events to the modified image
        self.modify_image_label.bind("<Button-1>", self.modify_pixel)
        self.modify_image_label.bind("<B1-Motion>", self.modify_pixel)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = load_image(file_path)
            self.pattern = image_to_pattern(self.image)
            self.show_image(self.image, self.image_label)
            self.modified_image = self.image.copy()
            self.show_image(self.modified_image, self.modify_image_label)

    def reload_image(self):
        if hasattr(self, 'modified_image'):
            self.image = self.modified_image.copy()
            self.pattern = image_to_pattern(self.image)
            self.show_image(self.image, self.image_label)
            messagebox.showinfo("Cargar la imagen nuevamente", "Se modifico la imagen exitosamente")
        else:
            messagebox.showerror("Error", "No modified image available to reload.")

    def show_image(self, image, label):
        image_pil = Image.fromarray(image)
        scaled_image_pil = image_pil.resize((500, 500), Image.NEAREST)  # Escalar la imagen para visualización
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
        else:
            messagebox.showerror("Error", "Ninguna imagen se subió")

    def modify_pixel(self, event):
        if hasattr(self, 'modified_image'):
            x = event.x * 100 // self.modify_image_label.winfo_width()
            y = event.y * 100 // self.modify_image_label.winfo_height()
            if 0 <= x < 100 and 0 <= y < 100:  # Ensure x and y are within bounds
                self.modified_image[y, x] = 255 if self.modified_image[y, x] == 0 else 0
                self.show_image(self.modified_image, self.modify_image_label)

root = tk.Tk()
app = ARTApp(root)
root.mainloop()
