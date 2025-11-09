import cv2
from utils import ginput

from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "paisaje_2.tiff"
src = cv2.imread(image_path)

# show image
cv2.imshow('image', src)

# graphic input of 5 points
pts = ginput('image', src, 6)

# print the result
print(pts)
cv2.waitKey(0)
cv2.destroyAllWindows()