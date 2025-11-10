import cv2
import numpy as np 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "aloel.jpg"
src = cv2.imread(image_path)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# Separar los tres canales H, S y V
H, S, V = cv2.split(hsv)

# Mostrar los canales HSV en escala de grises
cv2.imshow('Canal H (Tono / Hue)', H)
cv2.imshow('Canal S (Saturacion)', S)
cv2.imshow('Canal V (Valor / Brillo)', V)

# Esperar a que se presione una tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()