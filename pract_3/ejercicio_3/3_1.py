import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path 

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
img = np.asarray(Image.open(image_path))

def imadjust(img, low_in, high_in):
    min = low_in
    max = high_in
    img_out = np.round(255.0 * (img - min) / (max - min + 1)).astype(np.uint8)
    img_out[img < min] = 0
    img_out[img > max] = 255
    return img_out

# Ajustar el contraste con min=100 y max=200
img_contraste = imadjust(img, 100, 200)

# Mostrar imagen original y ajustada
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_contraste, cmap='gray')
plt.title('Contraste ajustado (100-200)')
plt.axis('off')

plt.tight_layout()
plt.show()