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

# Convertir a float para operar 
cam = np.array(cameraman_resized, dtype=float) 
moon = np.array(moon_resized, dtype=float) 

# ---- Combinación lineal ---- 
S = 1.8 * cam - 1.2 * moon + 128 

# Asegurar que los valores estén en rango válido [0,255] 
S = np.clip(S, 0, 255) 

# Mostrar imágenes originales y resultado 
fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 

axes[0].imshow(cam, cmap='gray') 
axes[0].set_title("Cameraman (256x256)") 
axes[0].axis('off') 

axes[1].imshow(moon, cmap='gray') 
axes[1].set_title("Moon (256x256)") 
axes[1].axis('off') 

axes[2].imshow(S, cmap='gray') 
axes[2].set_title("S = 1.8*CAM - 1.2*MOON + 128") 
axes[2].axis('off') 

plt.tight_layout() 

plt.show() 

 