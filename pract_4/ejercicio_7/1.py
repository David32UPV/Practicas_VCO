import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def label2rgb(label_img, label_ids):
    """Convierte los labels en una imagen a color para visualización"""
    # Mapear los labels a un tono HSV
    label_hue = np.uint8(179 * (label_img) / np.max(label_img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    
    # Convertir HSV a BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
    # Poner el fondo a negro (label 0)
    labeled_img[label_ids == 0] = 0
    return labeled_img

# Ventanas
window_name = 'Binary Image'
window_colored_name = 'Colored Objects'
cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow(window_colored_name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Parámetros
areaMinima = 300
umbral = 155
folders = './output1/'

# Selección de carpeta
root = tk.Tk()
root.withdraw()
folder_name = filedialog.askdirectory(initialdir=folders)

f2Name = folder_name + '/'
list_files = os.scandir(f2Name)

for ent in list_files:
    if ent.is_file() and ent.name.endswith('.jpg'):
        filename = f2Name + ent.name
        img = cv2.imread(filename)

        if img is None:
            print(f"No se pudo abrir {filename}")
            continue

        # Convertir a gris y aplicar umbral
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray_img, umbral, 255, cv2.THRESH_BINARY_INV)[1]

        # Componentes conectadas (8-conectividad)
        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(threshold, 8, cv2.CV_32S)

        # Filtrar por área mínima
        output_binary = np.zeros(gray_img.shape, dtype="uint8")
        for i in range(1, totalLabels):  # saltamos el fondo
            area = values[i, cv2.CC_STAT_AREA]
            if area > areaMinima:
                componentMask = (label_ids == i).astype("uint8") * 255
                output_binary = cv2.bitwise_or(output_binary, componentMask)

        # Convertir los labels filtrados a imagen coloreada
        # Para esto hacemos un label_ids filtrado: 0 donde área < areaMinima
        label_filtered = label_ids.copy()
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            if area <= areaMinima:
                label_filtered[label_ids == i] = 0

        output_colored = label2rgb(label_filtered, label_filtered)

        # Mostrar ventanas
        cv2.imshow(window_name, output_binary)
        cv2.imshow(window_colored_name, output_colored)

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # salir con 'q' o ESC
            break

cv2.destroyAllWindows()
