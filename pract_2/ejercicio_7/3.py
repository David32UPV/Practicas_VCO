import cv2
import numpy as np 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "aloel.jpg"
src = cv2.imread(image_path)

lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)

# Separar los tres canales L, a, b
L, a, b = cv2.split(lab)

# Mostrar los canales en escala de grises
cv2.imshow('Canal L (Luminosidad)', L)
cv2.imshow('Canal a (Verde - Rojo)', a)
cv2.imshow('Canal b (Azul - Amarillo)', b)

# Esperar a que se presione una tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()