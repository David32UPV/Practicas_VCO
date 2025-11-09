import cv2  # OpenCV
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "paisaje_2.tiff"

src = cv2.imread(image_path)
img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
dst = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)

plt.subplot(121),plt.imshow(img_rgb),plt.title('Input') # Visualización en matpolotlib.pyplot
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

#cv2.imshow('Imagen original', src)
#cv2.imshow('Imagen escalada', dst)

#cv2.waitKey(0)
#cv2.destroyAllWindows()