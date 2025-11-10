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

# Introducir ruido Salt & Pepper
src_salt_pepper = saltAndPepper_noise(src, 0.05)

# Introducir ruido Gaussiano
src_gaussian = gaussian_noise(src, mean=0, sigma=20)

# Aplicar filtrado Gaussiano a la imagen con ruido
filtered_gaussian = cv2.GaussianBlur(src_gaussian, (5, 5), 1.0)

# Crear figura con todas las imágenes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(src)
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(src_gaussian)
axs[1].set_title('Gaussian Noise (sigma=20)')
axs[1].axis('off')

axs[2].imshow(filtered_gaussian)
axs[2].set_title('Gaussian Blur Filtrado\n(ksize=5, sigma=1.0)')
axs[2].axis('off')

plt.tight_layout()
plt.show()  