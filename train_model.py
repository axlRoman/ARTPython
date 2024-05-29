import os
import pickle
from art import ART1
from image_processing import load_image, image_to_pattern

# Ruta a la carpeta principal
data_dir = "clases"

# Crear el modelo ART1
num_features = 10000  # Tamaño de la imagen 100x100
vigilance = 0.8
model = ART1(num_features, vigilance)

# Cargar y procesar las imágenes
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.endswith('.jpg'):
                image = load_image(img_path)
                pattern = image_to_pattern(image)
                model.train([pattern])
                print(f"Entrenado con {img_name} en la clase {class_name}")

# Guardar el modelo entrenado
# Guardar el modelo entrenado
with open('art_model.pkl', 'wb') as f:
    pickle.dump({'weights': model.weights}, f)

print("Modelo entrenado y guardado como art_model.pkl")