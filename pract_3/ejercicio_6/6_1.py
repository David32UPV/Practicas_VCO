import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

# Convertir a escala de grises para detección de bordes
src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

# Aplicar Sobel en dirección X (bordes verticales)
# ddepth=cv2.CV_32F para preservar valores negativos
# dx=1, dy=0 indica derivada en X
# ksize=3 es el tamaño del kernel
sobel_x = cv2.Sobel(src_gray, cv2.CV_32F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)  # convertir a uint8 y valores absolutos

# Aplicar Sobel en dirección Y (bordes horizontales)
# dx=0, dy=1 indica derivada en Y
sobel_y = cv2.Sobel(src_gray, cv2.CV_32F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)  # convertir a uint8 y valores absolutos

#crear ventana con la imagen original y los resultados
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(src_gray, cmap='gray')
axs[0].set_title('Original (Escala de grises)')
axs[0].axis('off')

axs[1].imshow(sobel_x, cmap='gray')
axs[1].set_title('Sobel X (Bordes Verticales)')
axs[1].axis('off')

axs[2].imshow(sobel_y, cmap='gray')
axs[2].set_title('Sobel Y (Bordes Horizontales)')
axs[2].axis('off')

plt.tight_layout()
plt.show()