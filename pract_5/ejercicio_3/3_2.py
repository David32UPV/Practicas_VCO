import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path 

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "nemo0.jpg"
nemo = cv2.imread(str(image_path))

# Convertir de BGR a RGB para que Matplotlib lo muestre correctamente
nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

# Convertir de BGR a HSV
nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)

# ============ SEGMENTACIÓN DE COLOR NARANJA ============
# Definir rango de color naranja en HSV
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

# Segmentar naranja usando inRange() function
mask_orange = cv2.inRange(nemo_hsv, light_orange, dark_orange)

# Bitwise-AND mask y imagen original para naranja
result_orange = cv2.bitwise_and(nemo, nemo, mask=mask_orange)

# Convertir resultado a RGB para mostrar correctamente
result_orange_rgb = cv2.cvtColor(result_orange, cv2.COLOR_BGR2RGB)

# ============ SEGMENTACIÓN DE COLOR BLANCO ============
# Definir rango de color blanco en HSV
light_white = (0, 0, 200)
dark_white = (145, 60, 255)

# Segmentar blanco usando inRange() function
mask_white = cv2.inRange(nemo_hsv, light_white, dark_white)

# Bitwise-AND mask y imagen original para blanco
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

# Convertir resultado a RGB para mostrar correctamente
result_white_rgb = cv2.cvtColor(result_white, cv2.COLOR_BGR2RGB)

# ============ COMBINAR MÁSCARAS ============
# Combinar ambas máscaras (suma de máscaras)
final_mask = cv2.add(mask_orange, mask_white)

# Aplicar máscara final
final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)

# Convertir resultado a RGB para mostrar correctamente
final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

# ============ MOSTRAR RESULTADOS ============
# Mostrar máscara y resultado de naranja y blanco
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Primera fila: Segmentación de naranja
axs[0, 0].imshow(nemo_rgb)
axs[0, 0].set_title("Original")
axs[0, 0].axis('off')

axs[0, 1].imshow(mask_orange, cmap="gray")
axs[0, 1].set_title("Máscara de Color Naranja")
axs[0, 1].axis('off')

axs[0, 2].imshow(result_orange_rgb)
axs[0, 2].set_title("Nemo Segmentado (Solo Naranja)")
axs[0, 2].axis('off')

# Segunda fila: Segmentación de blanco
axs[1, 0].imshow(nemo_rgb)
axs[1, 0].set_title("Original")
axs[1, 0].axis('off')

axs[1, 1].imshow(mask_white, cmap="gray")
axs[1, 1].set_title("Máscara de Color Blanco")
axs[1, 1].axis('off')

axs[1, 2].imshow(result_white_rgb)
axs[1, 2].set_title("Nemo Segmentado (Solo Blanco)")
axs[1, 2].axis('off')

plt.suptitle("Segmentación de Colores en Nemo (Naranja y Blanco)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============ MOSTRAR MÁSCARAS COMBINADAS ============
fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))

# Mostrar máscaras individuales y combinada
axs2[0].imshow(mask_orange, cmap="gray")
axs2[0].set_title("Máscara Naranja")
axs2[0].axis('off')

axs2[1].imshow(mask_white, cmap="gray")
axs2[1].set_title("Máscara Blanco")
axs2[1].axis('off')

axs2[2].imshow(final_mask, cmap="gray")
axs2[2].set_title("Máscara Combinada (Naranja + Blanco)")
axs2[2].axis('off')

plt.suptitle("Combinación de Máscaras", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============ MOSTRAR RESULTADO FINAL ============
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))

axs3[0].imshow(nemo_rgb)
axs3[0].set_title("Original")
axs3[0].axis('off')

axs3[1].imshow(final_result_rgb)
axs3[1].set_title("Nemo Segmentado (Naranja + Blanco)")
axs3[1].axis('off')

plt.suptitle("Resultado Final de Segmentación Combinada", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
