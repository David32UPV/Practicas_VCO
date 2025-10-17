from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "g3.png"
img = Image.open(image_path)

# 2. Convertir la imagen a escala de grises
# El modo 'L' representa una imagen en escala de grises de 8 bits
img_gris = img.convert('L')

# 3. Crear una figura con dos subplots lado a lado
plt.figure(figsize=(10, 5))

# 4. Mostrar la imagen original
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, posición 1
plt.imshow(img)
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes

# 5. Mostrar la imagen en escala de grises
plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, posición 2
plt.imshow(img_gris, cmap='gray')  # usar mapa de colores gris
plt.title('Imagen en Escala de Grises')
plt.axis('off')  # Ocultar los ejes

# 6. Ajustar el diseño y mostrar las imágenes
plt.tight_layout()
plt.show()