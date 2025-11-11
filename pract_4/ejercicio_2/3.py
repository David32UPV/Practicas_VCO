import cv2
import numpy as np
from tkinter import filedialog
import os

# Selección de una carpeta
path = filedialog.askdirectory(initialdir="./../", title="Seleccione una carpeta")

window_original = 'Original_image'
window_otsu = 'Otsu_threshold'
window_triangle = 'Triangle_threshold'

cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_otsu, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_triangle, cv2.WINDOW_NORMAL)

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        
        filename = os.path.join(root, name)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(window_original, img)

        # --- Método de Otsu ---
        ret_otsu, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Umbral Otsu para {name}: {ret_otsu:.2f}")
        cv2.imshow(window_otsu, thresh_otsu)

        # --- Método de Triangle ---
        ret_triangle, thresh_triangle = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        print(f"Umbral Triangle para {name}: {ret_triangle:.2f}")
        cv2.imshow(window_triangle, thresh_triangle)
        
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # Salir con 'q' o ESC
            break

cv2.destroyAllWindows()
