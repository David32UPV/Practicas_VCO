from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "aloel.jpg"
img = Image.open(image_path)

# Hacer la imagen mas pequeña para que quepa mejor
img = img.resize((400, 400))

# 2. Separar los canales RGB
# Convertir la imagen a un array de numpy
img_array = np.array(img)

# Extraer cada canal
canal_r = img_array[:,:,0]  # Canal Rojo
canal_g = img_array[:,:,1]  # Canal Verde
canal_b = img_array[:,:,2]  # Canal Azul

# 3. Crear una figura con cuatro subplots (2x2)
plt.figure(figsize=(12, 10))

# 4. Mostrar la imagen original
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Imagen Original')
plt.axis('off')

# 5. Mostrar canal Rojo
plt.subplot(2, 2, 2)
plt.imshow(canal_r, cmap='gray')
plt.title('Canal Rojo')
plt.axis('off')

# 6. Mostrar canal Verde
plt.subplot(2, 2, 3)
plt.imshow(canal_g, cmap='gray')
plt.title('Canal Verde')
plt.axis('off')

# 7. Mostrar canal Azul
plt.subplot(2, 2, 4)
plt.imshow(canal_b, cmap='gray')
plt.title('Canal Azul')
plt.axis('off')

# 8. Ajustar el diseño y mostrar
plt.tight_layout()
plt.show()
