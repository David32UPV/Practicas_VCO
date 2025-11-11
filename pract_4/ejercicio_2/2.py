import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path 

# Obtener ruta relativa a la imagen 
image_path = Path(__file__).parent.parent / "Test" / "IMG_20210321_191546.jpg" 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

# Aplicar los diferentes tipos de umbralización
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)


titles = ['Original Image','BINARY','BINARY_INV']
images = [img, thresh1, thresh2]
for i in range(3):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()