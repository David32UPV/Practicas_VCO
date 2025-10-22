import cv2
import numpy as np
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "paisaje_2.tiff"
src = cv2.imread(image_path)

rows, cols = src.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(src, M, (rows + 100, cols + 50))

cv2.imshow('Imagen original', src)
cv2.imshow('Imagen trasladada', dst)

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
dst2 = cv2.warpAffine(src, M, (cols, rows))

cv2.imshow('Imagen rotada', dst2)

# Escalar la imagen
M = cv2.scaleAdd(src, )
cv2.waitKey()
cv2.destroyAllWindows()