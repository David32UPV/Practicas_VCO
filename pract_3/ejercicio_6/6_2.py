import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

# Convertir a escala de grises
src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

# Método 1: LoG tradicional (Laplacian of Gaussian)
# Primero aplicar Gaussian Blur
gaussian_blur = cv2.GaussianBlur(src_gray, (5, 5), 1.0)

# Luego aplicar Laplacian
laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_32F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Crear figura con comparación
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(src_gray, cmap='gray')
axs[0].set_title('Original (Escala de grises)')
axs[1].imshow(gaussian_blur, cmap='gray')
axs[1].set_title('Gaussian Blur (σ=1.0)')
axs[2].imshow(laplacian, cmap='gray')
axs[2].set_title('LoG (Gaussian + Laplacian)')


plt.tight_layout()
plt.show()
