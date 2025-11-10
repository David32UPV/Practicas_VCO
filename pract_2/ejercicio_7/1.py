import cv2 
import numpy as np  
from pathlib import Path 

# Obtener ruta relativa a la imagen 
image_path = Path(__file__).parent.parent / "images" / "aloel.jpg" 
src = cv2.imread(image_path) 

# Separar canales B, G y R 
B, G, R = cv2.split(src) 

# Mostrar los canales como imágenes en escala de grises 
cv2.imshow('Canal R (Rojo)', R) 
cv2.imshow('Canal G (Verde)', G) 
cv2.imshow('Canal B (Azul)', B) 

# Esperar a que se presione una tecla y cerrar ventanas 
cv2.waitKey(0) 
cv2.destroyAllWindows() 