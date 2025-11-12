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

# Definir rangos de color en HSV
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)
light_white = (0, 0, 200)
dark_white = (145, 60, 255)

# Almacenar imágenes originales y segmentadas
original_images = []
segmented_images = []

for nemo_file in nemo_files:
    # Leer imagen
    nemo = cv2.imread(str(nemo_file))
    
    # Convertir de BGR a RGB
    nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    original_images.append(nemo_rgb)
    
    # Convertir de BGR a HSV
    nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
    
    # ============ SEGMENTACIÓN DE COLOR NARANJA ============
    mask_orange = cv2.inRange(nemo_hsv, light_orange, dark_orange)
    
    # ============ SEGMENTACIÓN DE COLOR BLANCO ============
    mask_white = cv2.inRange(nemo_hsv, light_white, dark_white)
    
    # ============ COMBINAR MÁSCARAS ============
    final_mask = cv2.add(mask_orange, mask_white)
    
    # Aplicar máscara final
    final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
    
    # Convertir resultado a RGB
    final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    segmented_images.append(final_result_rgb)

# Crear figura con 2 filas y tantas columnas como imágenes haya
fig, axs = plt.subplots(2, len(nemo_files), figsize=(16, 8))

# Si solo hay una imagen, axs no es una matriz bidimensional
if len(nemo_files) == 1:
    axs = axs.reshape(2, 1)

# Primera fila: Imágenes originales
for idx, original_img in enumerate(original_images):
    axs[0, idx].imshow(original_img)
    axs[0, idx].set_title(f"Original - {nemo_files[idx].name}")
    axs[0, idx].axis('off')

# Segunda fila: Imágenes segmentadas
for idx, segmented_img in enumerate(segmented_images):
    axs[1, idx].imshow(segmented_img)
    axs[1, idx].set_title(f"Segmentado - {nemo_files[idx].name}")
    axs[1, idx].axis('off')

plt.suptitle("Segmentación de Colores Naranja y Blanco en 6 Imágenes de Nemo", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
