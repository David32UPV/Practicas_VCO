import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

def clamp(num, min_value=0, max_value=255):
    """Limita un valor entre min y max"""
    return int(max(min(num, max_value), min_value))

def saltAndPepper_noise(img, percent):
    """
    Introduce ruido Salt & Pepper a la imagen
    img: Imagen a la que introducir el ruido
    percent: [0,1) porcentaje de ruido
    """
    img_copy = img.copy()
    per = int(percent * img_copy.size)
    for k in range(per):
        i = int(np.random.random() * img_copy.shape[1])
        j = int(np.random.random() * img_copy.shape[0])
        if img_copy.ndim == 2:
            img_copy[j, i] = 255
        elif img_copy.ndim == 3:
            img_copy[j, i, 0] = 255
            img_copy[j, i, 1] = 255
            img_copy[j, i, 2] = 255
    return img_copy

def gaussian_noise(img, mean=0, sigma=20):
    """
    Introduce ruido Gaussiano a la imagen
    img: Imagen a la que introducir el ruido
    mean: Media del ruido gaussiano
    sigma: Desviación estándar del ruido gaussiano
    """
    img_copy = img.copy().astype(np.float32)
    h, w, c = img_copy.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(mean, sigma, 3)
            img_copy[row, col, 0] = clamp(img_copy[row, col, 0] + s[0])
            img_copy[row, col, 1] = clamp(img_copy[row, col, 1] + s[1])
            img_copy[row, col, 2] = clamp(img_copy[row, col, 2] + s[2])
    return img_copy.astype(np.uint8)

# Introducir ruido Gaussiano
src_gaussian = gaussian_noise(src, mean=0, sigma=20)

# Aplicar filtrado Gaussiano a la imagen con ruido
filtered_gaussian_blur = cv2.GaussianBlur(src_gaussian, (5, 5), 1.0)

# Aplicar filtrado de Mediana a la imagen con ruido
filtered_median = cv2.medianBlur(src_gaussian, 5)

# Crear figura con todas las imágenes
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Primera fila: Imagen original y con ruido
axs[0, 0].imshow(src)
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[0, 1].imshow(src_gaussian)
axs[0, 1].set_title('Gaussian Noise (sigma=20)')
axs[0, 1].axis('off')

# Segunda fila: Filtros aplicados
axs[1, 0].imshow(filtered_gaussian_blur)
axs[1, 0].set_title('Gaussian Blur Filtrado\n(ksize=5, sigma=1.0)')
axs[1, 0].axis('off')

axs[1, 1].imshow(filtered_median)
axs[1, 1].set_title('Median Blur Filtrado\n(ksize=5)')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()