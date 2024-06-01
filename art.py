import sys
import numpy as np
import cv2
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
                             QFileDialog, QSlider, QHBoxLayout, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

class ART:
    def __init__(self, rho=0.5, tolerancia=0.001):
        self.rho = rho  # Parámetro de vigilancia
        self.tolerancia = tolerancia  # Tolerancia para la similitud
        self.pesos = []

    def entrenar(self, patron, clase):
        patron = np.array(patron)
        if not self.pesos:
            self.pesos.append((patron, clase))
        else:
            mejor_similitud = -1
            indice_mejor_peso = -1
            for i, (peso, _) in enumerate(self.pesos):
                similitud = self._calcular_similitud(patron, peso)
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    indice_mejor_peso = i
            if mejor_similitud >= self.rho - self.tolerancia:
                peso, clase_existente = self.pesos[indice_mejor_peso]
                self._actualizar_peso(patron, peso, mejor_similitud, indice_mejor_peso)
            else:
                self.pesos.append((patron, clase))
        self.guardar_pesos('art_model.pkl')  # Guardar después de entrenar

    def predecir(self, patron):
        mejor_similitud = -1
        mejor_clase = None
        mejor_peso = None
        patron = np.array(patron)
        for peso, clase in self.pesos:
            similitud = self._calcular_similitud(patron, peso)
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_clase = clase
                mejor_peso = peso

        if mejor_similitud >= self.rho - self.tolerancia:
            return mejor_clase, mejor_peso
        else:
            return None, None

    def _actualizar_peso(self, patron, peso, similitud, indice_mejor_peso):
        alpha = 0.5  # Factor de aprendizaje
        peso_nuevo = peso + alpha * (patron - peso)
        self.pesos[indice_mejor_peso] = (peso_nuevo, self.pesos[indice_mejor_peso][1])

    def _calcular_similitud(self, patron1, patron2):
        patron1_norm = patron1 / np.linalg.norm(patron1)
        patron2_norm = patron2 / np.linalg.norm(patron2)
        return np.dot(patron1_norm, patron2_norm)

    def guardar_pesos(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.pesos, f)

    def cargar_pesos(self, filename):
        with open(filename, 'rb') as f:
            self.pesos = pickle.load(f)

class DrawingArea(QLabel):
    def __init__(self, parent=None):
        super(DrawingArea, self).__init__(parent)
        self.setFixedSize(400, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = None
        self.pen_width = 10  # Ancho del pincel

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        return self.image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ART Symbol Recognition')
        self.art = ART()
        self.initUI()
        self.load_art_model()

    def initUI(self):
        # Crear el área de dibujo
        self.drawing_area = DrawingArea(self)

        # Crear el área de resultados
        self.result_label = QLabel('Clase Predicha:', self)
        self.result_label.setFixedSize(400, 50)

        # Crear el área de la imagen más similar
        self.similar_image_label = QLabel('Imagen Similar:', self)
        self.similar_image_display = QLabel(self)
        self.similar_image_display.setFixedSize(400, 400)

        # Crear los botones y el slider
        self.upload_button = QPushButton('Subir Imagen', self)
        self.upload_button.clicked.connect(self.upload_image)

        self.predict_button = QPushButton('Clasificar', self)
        self.predict_button.clicked.connect(self.predict)

        self.clear_button = QPushButton('Limpiar', self)
        self.clear_button.clicked.connect(self.drawing_area.clear)

        self.vigilance_slider = QSlider(Qt.Horizontal, self)
        self.vigilance_slider.setRange(0, 100)
        self.vigilance_slider.setValue(int(self.art.rho * 100))
        self.vigilance_slider.valueChanged.connect(self.update_vigilance)

        self.vigilance_label = QLabel(f'Parámetro de vigilancia: {self.art.rho}', self)

        # Crear los layouts
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.predict_button)
        buttons_layout.addWidget(self.clear_button)

        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drawing_area)
        left_layout.addLayout(buttons_layout)
        left_layout.addWidget(self.vigilance_slider)
        left_layout.addWidget(self.vigilance_label)
        left_layout.addWidget(self.result_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.similar_image_label)
        right_layout.addWidget(self.similar_image_display)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (400, 400))
            self.drawing_area.image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
            self.drawing_area.update()

    def predict(self):
        image = self.drawing_area.get_image()
        image_array = self.qimage_to_array(image)
        image_flat = image_array.flatten()
        clase, similar_peso = self.art.predecir(image_flat)

        if clase is None:
            text, ok = QInputDialog.getText(self, 'Nueva Clase', 'Introduce el nombre de la nueva clase:')
            if ok:
                self.art.entrenar(image_flat, text)
                self.result_label.setText(f'Clase Predicha: {text} (Nueva clase añadida)')
        else:
            self.result_label.setText(f'Clase Predicha: {clase}')
            similar_image = QImage(similar_peso.reshape(400, 400).astype(np.uint8), 400, 400, QImage.Format_Grayscale8)
            self.similar_image_display.setPixmap(QPixmap.fromImage(similar_image))

    def update_vigilance(self, value):
        self.art.rho = value / 100.0
        self.vigilance_label.setText(f'Parámetro de vigilancia: {self.art.rho}')

    def qimage_to_array(self, image):
        image = image.convertToFormat(QImage.Format_Grayscale8)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        array = np.frombuffer(ptr, np.uint8).reshape((height, width))
        return array

    def load_art_model(self):
        try:
            self.art.cargar_pesos('art_model.pkl')
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
