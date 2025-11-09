from skimage import data 
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import numpy as np 

# Cargar imágenes de ejemplo 
cameraman = data.camera() # imagen equivalente a cameraman.tif 
moon = data.moon() # imagen equivalente a moon.tif 

# Redimensionar ambas a 256x256 
target_shape = (256, 256) 
cameraman_resized = resize(cameraman, target_shape, anti_aliasing=True) 
moon_resized = resize(moon, target_shape, anti_aliasing=True) 

cam = np.array(cameraman_resized, dtype=float) 
moon = np.array(moon_resized, dtype=float) 

# Operaciones aritméticas 
img_sum = (cam + moon) / 2 # promedio (mezcla de ambas) 
img_diff = np.abs(cam - moon) # diferencia absoluta 
img_mult = cam * moon # producto píxel a píxel 
img_ratio = cam / (moon + 1e-6) # cociente (evitando dividir por cero) 

# Mostrar resultados 
fig, axes = plt.subplots(2, 3, figsize=(12, 8)) 

axes[0,0].imshow(cam, cmap='gray') 
axes[0,0].set_title("Cameraman (256x256)") 
axes[0,0].axis('off') 

axes[0,1].imshow(moon, cmap='gray') 
axes[0,1].set_title("Moon (256x256)") 
axes[0,1].axis('off') 

axes[0,2].imshow(img_sum, cmap='gray') 
axes[0,2].set_title("Promedio (cam + moon)/2") 
axes[0,2].axis('off') 

axes[1,0].imshow(img_diff, cmap='gray') 
axes[1,0].set_title("Diferencia |cam - moon|") 
axes[1,0].axis('off') 

axes[1,1].imshow(img_mult, cmap='gray') 
axes[1,1].set_title("Producto (cam * moon)") 
axes[1,1].axis('off') 

axes[1,2].imshow(img_ratio, cmap='gray') 
axes[1,2].set_title("Cociente (cam / moon)") 
axes[1,2].axis('off') 

plt.tight_layout() 

plt.show() 