import cv2
import numpy as np 
from utils import ginput

from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "building5.JPG"
src = cv2.imread(image_path)

# show image
cv2.imshow('image', src)

# graphic input of 4 points
pts = ginput('image', src, 5)
pts = np.array(pts, dtype=np.float32)  # convertir a float32
print(pts)

# Dibujar polígono sobre los puntos seleccionados
pts_int = pts.astype(int)
cv2.polylines(src, [pts_int], isClosed=True, color=(0,255,0), thickness=2)
cv2.imshow('Original with polygon', src)

outs = [[0,0], [200,0], [200,200], [0,200]]
outs = np.array(outs,np.float32)
print(outs)

M = cv2.getPerspectiveTransform(pts,outs)
dst = cv2.warpPerspective(src, M, (200, 200))

cv2.imshow('rectified', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()