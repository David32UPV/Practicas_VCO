import matplotlib.colors as mcolors 
import matplotlib.pyplot as plt 
import numpy as np  
import string 
from PIL import Image
import math 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "Paisaje_de_Albacete.jpg"
img = Image.open(image_path)

img.size # (640, 480) 

# Obtener la imagen transformada 
img_t = img 
plt.figure(figsize=(10, 6)) 
plt.subplot(121) 
plt.imshow(img) 

#plt.colorbar() 
plt.subplot(122) 
plt.imshow(np.asarray(img_t)) 

#plt.colorbar() 

plt.show() 