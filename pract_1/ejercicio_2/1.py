from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "g3.png"
img = Image.open(image_path)

#Probar transformaciones
# Rotar 90 grados
img_rotated = img.rotate(90)

# Volteo horizontal
img_flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)

# Volteo vertical 
img_flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)

# Mostrar las imágenes originales y transformadas
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_rotated)
plt.title("Rotada 90 grados")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img_flipped_h)
plt.title("Volteada horizontal")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(img_flipped_v)
plt.title("Volteada vertical")
plt.axis("off")

plt.show()