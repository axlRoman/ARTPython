import os
import pickle
import numpy as np
from art import ART1
from image_processing import load_image, image_to_pattern

# Ruta a la carpeta principal
data_dir = "clases"

# Función para evaluar el modelo
def evaluate_model(model, test_data):
    correct_predictions = 0
    total_predictions = 0
    for class_name, patterns in test_data.items():
        for pattern, label in patterns:
            prediction = model.predict(pattern)
            if prediction == class_name:
                correct_predictions += 1
            total_predictions += 1
    return correct_predictions / total_predictions

# Crear y entrenar el modelo ART1 con diferentes valores de vigilancia
vigilance_values = [0.6, 0.7, 0.8, 0.9]
best_vigilance = 0.8
best_accuracy = 0
num_features = 10000  # Tamaño de la imagen 100x100

# Cargar y procesar las imágenes
training_data = {}
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        training_data[class_name] = []
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.endswith('.jpg'):
                image = load_image(img_path)
                pattern = image_to_pattern(image)
                training_data[class_name].append((pattern, class_name))

# Dividir los datos en entrenamiento y prueba
train_data = {}
test_data = {}
for class_name, patterns in training_data.items():
    split_point = int(0.8 * len(patterns))
    train_data[class_name] = patterns[:split_point]
    test_data[class_name] = patterns[split_point:]

# Probar diferentes valores de vigilancia
for vigilance in vigilance_values:
    model = ART1(num_features, vigilance)
    for class_name, patterns in train_data.items():
        for pattern, label in patterns:
            model.train([pattern])
    accuracy = evaluate_model(model, test_data)
    print(f"Vigilance: {vigilance}, Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_vigilance = vigilance

print(f"Best vigilance: {best_vigilance} with accuracy: {best_accuracy}")

# Entrenar el mejor modelo
model = ART1(num_features, best_vigilance)
for class_name, patterns in train_data.items():
    for pattern, label in patterns:
        model.train([pattern])

# Guardar el modelo entrenado
with open('art_model.pkl', 'wb') as f:
    pickle.dump({'weights': model.weights}, f)

print("Modelo entrenado y guardado como art_model.pkl")
