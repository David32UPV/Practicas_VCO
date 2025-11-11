# umbralizacon_adaptativa.py
#
# Programa para realizar operaciones de umbralización local con imágenes de niveles de gris.
#
# Autor: José M Valiente    Fecha: noviembre 2025 (modificado)
#
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
cv2.namedWindow(window_original, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# Tamaño de ventana inicial para la adaptativa
low_H = 120
method_adaptive = 'midgrey'   # Cambiar a 'midgrey' si se desea

def myAdaptiveThreshols(img, method='midgrey', block_size=15, k=-0.2):
    # Convertir a gris si es color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Asegurar block_size impar >=3
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    if method.lower() == 'midgrey':
        # Umbral = (mínimo local + máximo local)/2
        min_img = cv2.erode(gray, np.ones((block_size, block_size), np.uint8))
        max_img = cv2.dilate(gray, np.ones((block_size, block_size), np.uint8))
        thresh_img = ((min_img.astype(np.float32) + max_img.astype(np.float32)) / 2).astype(np.uint8)
        binary = np.where(gray >= thresh_img, 255, 0).astype(np.uint8)

    elif method.lower() == 'niblack':
        # Media y desviación estándar local
        mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(block_size, block_size))
        sq_mean = cv2.boxFilter((gray.astype(np.float32)**2), ddepth=-1, ksize=(block_size, block_size))
        std = np.sqrt(sq_mean - mean**2)
        thresh_img = mean + k*std
        binary = np.where(gray >= thresh_img, 255, 0).astype(np.uint8)

    else:
        raise ValueError("method debe ser 'midgrey' o 'niblack'")

    return binary

def on_thresh_trackbar(val):
    global low_H, window_threshold, img
    low_H = max(3, val | 1)  # asegurar impar >=3
    cv2.setTrackbarPos('trackbar', window_threshold, low_H)
    thresh1 = myAdaptiveThreshols(img, method=method_adaptive, block_size=low_H, k=-0.2)
    cv2.imshow(window_threshold, thresh1)

# Selección de una carpeta mediante un diálogo
path = filedialog.askdirectory(initialdir="./../", title="Seleccione una carpeta")

# Crear barra de deslizamiento (trackbar)
cv2.createTrackbar('trackbar', window_threshold, low_H, 255, on_thresh_trackbar)

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        filename = os.path.join(root, name)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(window_original, img)

        # Mostrar umbralización adaptativa al cargar la imagen
        thresh1 = myAdaptiveThreshols(img, method=method_adaptive, block_size=low_H, k=-0.2)
        cv2.imshow(window_threshold, thresh1)

        key = -1
        while key == -1:
            key = cv2.pollKey()
            # No sobrescribimos thresh1 aquí; la trackbar lo actualizará

        if key == ord('q') or key == 27:  # 'q' o ESC para acabar
            break

cv2.destroyWindow(window_original)
cv2.destroyWindow(window_threshold)
