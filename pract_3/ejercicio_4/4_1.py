import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

# Kernel emboss
H = np.array([
 [0, -1, -1],
 [1,  0, -1],
 [1,  1,  0]
], dtype=np.float32)

# Aplicar filtro
dst = cv2.filter2D(src, -1, H)

# Mostrar en una sola ventana (figura) la original y la filtrada
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(src)
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(dst)
axs[1].set_title("Emboss Filter")
axs[1].axis('off')

plt.tight_layout()
plt.show()