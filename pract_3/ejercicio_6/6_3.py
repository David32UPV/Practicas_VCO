import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

# Convertir a escala de grises para detección de bordes
src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

# ============ VENTANA 1: SOBEL ============
# Aplicar Sobel en dirección X (bordes verticales)
sobel_x = cv2.Sobel(src_gray, cv2.CV_32F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# Aplicar Sobel en dirección Y (bordes horizontales)
sobel_y = cv2.Sobel(src_gray, cv2.CV_32F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combinar ambas direcciones
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

fig1, axs1 = plt.subplots(1, 4, figsize=(18, 5))
axs1[0].imshow(src_gray, cmap='gray')
axs1[0].set_title('Original (Escala de grises)')
axs1[0].axis('off')

axs1[1].imshow(sobel_x, cmap='gray')
axs1[1].set_title('Sobel X (Bordes Verticales)')
axs1[1].axis('off')

axs1[2].imshow(sobel_y, cmap='gray')
axs1[2].set_title('Sobel Y (Bordes Horizontales)')
axs1[2].axis('off')

axs1[3].imshow(sobel_combined, cmap='gray')
axs1[3].set_title('Sobel Combinado (Magnitud)')
axs1[3].axis('off')

fig1.suptitle('VENTANA 1: Detección de Bordes con SOBEL', fontsize=14, fontweight='bold')
plt.tight_layout()

# ============ VENTANA 2: LoG ============
# Método 1: LoG tradicional (Laplacian of Gaussian)
gaussian_blur = cv2.GaussianBlur(src_gray, (5, 5), 1.0)
laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_32F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Laplacian directo (sin Gaussian previo)
laplacian_direct = cv2.Laplacian(src_gray, cv2.CV_32F, ksize=3)
laplacian_direct = cv2.convertScaleAbs(laplacian_direct)

fig2, axs2 = plt.subplots(1, 4, figsize=(18, 5))
axs2[0].imshow(src_gray, cmap='gray')
axs2[0].set_title('Original (Escala de grises)')
axs2[0].axis('off')

axs2[1].imshow(gaussian_blur, cmap='gray')
axs2[1].set_title('Gaussian Blur (σ=1.0)')
axs2[1].axis('off')

axs2[2].imshow(laplacian_direct, cmap='gray')
axs2[2].set_title('Laplacian Directo')
axs2[2].axis('off')

axs2[3].imshow(laplacian, cmap='gray')
axs2[3].set_title('LoG (Gaussian + Laplacian)')
axs2[3].axis('off')

fig2.suptitle('VENTANA 2: Detección de Bordes con LoG', fontsize=14, fontweight='bold')
plt.tight_layout()

# ============ VENTANA 3: COMPARACIÓN GENERAL ============
# Scharr (más sensible)
scharr_x = cv2.Scharr(src_gray, cv2.CV_32F, 1, 0)
scharr_x = cv2.convertScaleAbs(scharr_x)

scharr_y = cv2.Scharr(src_gray, cv2.CV_32F, 0, 1)
scharr_y = cv2.convertScaleAbs(scharr_y)

scharr_combined = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

fig3, axs3 = plt.subplots(1, 4, figsize=(18, 5))
axs3[0].imshow(src_gray, cmap='gray')
axs3[0].set_title('Original (Escala de grises)')
axs3[0].axis('off')

axs3[1].imshow(sobel_combined, cmap='gray')
axs3[1].set_title('Sobel Combinado')
axs3[1].axis('off')

axs3[2].imshow(scharr_combined, cmap='gray')
axs3[2].set_title('Scharr Combinado')
axs3[2].axis('off')

axs3[3].imshow(laplacian, cmap='gray')
axs3[3].set_title('LoG')
axs3[3].axis('off')

fig3.suptitle('VENTANA 3: Comparación de Métodos de Detección de Bordes', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()