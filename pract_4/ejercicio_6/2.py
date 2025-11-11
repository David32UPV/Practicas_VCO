import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

window_name = 'Original image'
window_filtered_name = 'Filtered objects'
cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow(window_filtered_name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

areaMinima = 500
umbral = 155
folders = './output1/'
folder_name = filedialog.askdirectory(initialdir=folders)

f2Name = folder_name + '/'
list_files = os.scandir(f2Name)

for ent in list_files:
    if ent.is_file() and ent.name.endswith('.jpg'):
        filename = f2Name + ent.name
        img = cv2.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Umbralización
        threshold = cv2.threshold(gray_img, umbral, 255, cv2.THRESH_BINARY)[1]

        # Componentes conectadas (4-conectividad)
        analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        output = np.zeros(gray_img.shape, dtype="uint8")

        for i in range(1, totalLabels):  # saltar fondo (índice 0)
            area = values[i, cv2.CC_STAT_AREA]
            if area > areaMinima:
                # Máscara del componente
                componentMask = (label_ids == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)

                cX, cY = centroid[i]
                cX, cY = int(cX), int(cY)
                cv2.circle(img, (cX, cY), 4, (255, 255, 0), 5)  # amarillo (BGR), radio 4, grosor 5

        cv2.imshow(window_name, img)
        cv2.imshow(window_filtered_name, output)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # 'q' o ESC para salir
            break

cv2.destroyAllWindows()
