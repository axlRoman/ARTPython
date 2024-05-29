import cv2

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (100, 100))  # Redimensionar a 100x100 píxeles
    _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def image_to_pattern(image):
    return image.flatten() / 255
