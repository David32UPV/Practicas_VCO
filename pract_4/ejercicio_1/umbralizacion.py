import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path 

# Obtener ruta relativa a la imagen 
image_path = Path(__file__).parent.parent / "Test" / "IMG_20210321_122201.jpg" 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

# Aplicar los diferentes tipos de umbralización
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
ret, thresh6 = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
ret, thresh7 = cv2.threshold(img, 127, 255, cv2.THRESH_TRIANGLE)
ret, thresh8 = cv2.threshold(img, 127, 255, cv2.THRESH_MASK)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'OTSU', 'TRIANGLE', 'MASK']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7, thresh8]
for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()