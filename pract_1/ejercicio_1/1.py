from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "Paisaje_de_Albacete.jpg"
im = Image.open(image_path)

print(im.format, im.size, im.mode) 

img = np.asarray(im) 
imgplot = plt.imshow(img) 
plt.show() 