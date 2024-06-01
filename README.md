
# ARTPython

ARTPython es una implementación en Python de las redes neuronales de Teoría de la Resonancia Adaptativa (ART). Este proyecto tiene como objetivo mejorar el entrenamiento y la funcionalidad de los modelos ART, centrándose en la detección y clasificación de varios patrones, incluidos los símbolos aritméticos.

## Características

- **Detección de Patrones**: Detecta y clasifica varios patrones usando redes neuronales ART.
- **Gestión Dinámica de Clases**: Agrega nuevas clases dinámicamente durante la ejecución.
- **Interfaz de Visualización**: Una interfaz gráfica de usuario (GUI) fácil de usar para dibujar o cargar imágenes, ajustar parámetros de vigilancia y gestionar clases.
- **Persistencia**: Guarda y carga modelos y patrones entrenados.

## Instalación

1. Clona el repositorio:
   ```sh
   git clone https://github.com/axlRoman/ARTPython.git
   ```
2. Navega al directorio del proyecto:
   ```sh
   cd ARTPython
   ```
3. Instala las dependencias requeridas:
   ```sh
   pip install -r requirements.txt
   ```

## Uso

### Ejecutar la Aplicación

Para iniciar la aplicación, ejecuta:
```sh
python main.py
```

### Descripción de la Interfaz

- **Área de Dibujo**: Dibuja símbolos o patrones directamente.
- **Botón de Subir**: Sube una imagen desde tu dispositivo.
- **Botón de Clasificar**: Clasifica la imagen dibujada o cargada.
- **Deslizador de Vigilancia**: Ajusta el parámetro de vigilancia (rango: 0.0 a 1.0).
- **Botón de Limpiar**: Limpia el área de dibujo.
- **Agregar Nueva Clase**: Si un patrón no es reconocido, agrégalo como una nueva clase.
- **Guardar Patrones**: Guarda los patrones y el estado actual del modelo en un archivo.

### Entrenar el Modelo

1. Asegúrate de que tus imágenes de entrenamiento estén organizadas en la carpeta `clases`, con cada subcarpeta nombrada según la categoría (por ejemplo, `bulbasaur`, `pikachu`).
2. Ejecuta el script de entrenamiento:
   ```sh
   python train_model.py
   ```

## Estructura del Proyecto

- `main.py`: Script principal para ejecutar la aplicación.
- `train_model.py`: Script para entrenar el modelo ART.
- `gui.py`: Implementación de la GUI.
- `art_model.py`: Implementación del modelo ART.
- `clases/`: Carpeta que contiene imágenes de entrenamiento organizadas por categoría.
- `requirements.txt`: Lista de dependencias.
