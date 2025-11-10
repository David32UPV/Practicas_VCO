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

# Introducir ruido Salt & Pepper
src_salt_pepper = saltAndPepper_noise(src, 0.05)


# Crear figura con todas las imágenes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Primera fila
axs[0].imshow(src)
axs[0].set_title('Original')

axs[1].imshow(src_salt_pepper)
axs[1].set_title('Salt & Pepper Noise (5%)')

plt.tight_layout()
plt.show()