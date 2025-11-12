import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path 

# Obtener ruta de la carpeta de imágenes
images_path = Path(__file__).parent.parent / "images"

# Buscar todas las imágenes de nemo
nemo_files = sorted(images_path.glob("nemo*.jpg"))

print(f"Se encontraron {len(nemo_files)} imágenes de Nemo")

# Definir rango de color naranja en HSV
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

# Crear figura con 2 filas y tantas columnas como imágenes haya
fig, axs = plt.subplots(2, len(nemo_files), figsize=(16, 6))

# Si solo hay una imagen, axs no es una matriz
if len(nemo_files) == 1:
    axs = axs.reshape(2, 1)

for idx, nemo_file in enumerate(nemo_files):
    # Leer imagen
    nemo = cv2.imread(str(nemo_file))
    
    # Convertir de BGR a RGB
    nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    
    # Convertir de BGR a HSV
    nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
    
    # Segmentar naranja
    mask_orange = cv2.inRange(nemo_hsv, light_orange, dark_orange)
    
    # Aplicar máscara
    result = cv2.bitwise_and(nemo, nemo, mask=mask_orange)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Primera fila: imagen original
    axs[0, idx].imshow(nemo_rgb)
    axs[0, idx].set_title(f"Original {nemo_file.name}")
    axs[0, idx].axis('off')
    
    # Segunda fila: detección de naranja
    axs[1, idx].imshow(result_rgb)
    axs[1, idx].set_title(f"Naranja {nemo_file.name}")
    axs[1, idx].axis('off')

plt.suptitle("Detección de Color Naranja en 6 Imágenes de Nemo", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
