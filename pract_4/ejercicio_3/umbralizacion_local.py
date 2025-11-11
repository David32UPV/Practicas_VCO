# umbralizacon_global.py
#
# Programa pasa realizar operaciones de umbrallización global con imágenes de niveles de gris.
#
# Autor: José M Valiente    Fecha: marzo 2023
#
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import filedialog
import os

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
cv2.namedWindow(window_original, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# Tamaño de ventana inicial para la adaptativa
low_H = 120
method_adaptive = 'gaussian'  # Cambiar a 'mean' si se desea

def myAdaptiveThreshols(img, method='gaussian', block_size=15, C=2):
    # Convertir a gris si es color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Asegurar block_size impar y menor que tamaño de la imagen
    block_size = max(3, block_size | 1)
    block_size = min(block_size, min(gray.shape)//2)

    if method.lower() == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    thresh = cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C
    )
    return thresh

def on_thresh_trackbar(val):
    global low_H, window_threshold, img
    low_H = max(3, val | 1)  # asegurar impar >=3
    cv2.setTrackbarPos('trackbar', window_threshold, low_H)
    thresh1 = myAdaptiveThreshols(img, method=method_adaptive, block_size=low_H, C=2)
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
        thresh1 = myAdaptiveThreshols(img, method=method_adaptive, block_size=low_H, C=2)
        cv2.imshow(window_threshold, thresh1)

        key = -1
        while key == -1:
            key = cv2.pollKey()
            # La trackbar actualizará thresh1 dinámicamente

        if key == ord('q') or key == 27:  # 'q' o ESC para acabar
            break

cv2.destroyWindow(window_original)
cv2.destroyWindow(window_threshold)   

 