import numpy as np
import cv2 
from matplotlib import pyplot as plt
from pathlib import Path 
from PIL import Image

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "clahe_1.png"
img = np.asarray(Image.open(image_path))

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar ecualización del histograma
eq_gray = cv2.equalizeHist(gray)

# Aplicar CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)

# Calcular histogramas
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])
hist_clahe = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])

# Crear figura para las imágenes
plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(gray, cmap='gray')
plt.title('Escala de grises'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(eq_gray, cmap='gray')
plt.title('Ecualizada'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(clahe_img, cmap='gray')
plt.title('CLAHE'), plt.xticks([]), plt.yticks([])

# Crear figura para los histogramas
plt.figure(figsize=(12, 8))
plt.subplot(221)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
plt.title('Histograma Original')
plt.xlim([0,256])

plt.subplot(222)
plt.plot(hist_gray, color='gray')
plt.title('Histograma Escala de grises')
plt.xlim([0,256])

plt.subplot(223)
plt.plot(hist_eq, color='gray')
plt.title('Histograma Ecualizado')
plt.xlim([0,256])

plt.subplot(224)
plt.plot(hist_clahe, color='gray')
plt.title('Histograma CLAHE')
plt.xlim([0,256])

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()