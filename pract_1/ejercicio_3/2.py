from skimage import data 
from skimage.transform import resize 
import matplotlib.pyplot as plt 

# Cargar imágenes de ejemplo 
cameraman = data.camera() # imagen equivalente a cameraman.tif 
moon = data.moon() # imagen equivalente a moon.tif 

# Redimensionar ambas a 256x256 
target_shape = (256, 256) 
cameraman_resized = resize(cameraman, target_shape, anti_aliasing=True) 
moon_resized = resize(moon, target_shape, anti_aliasing=True) 

# Mostrar ambas redimensionadas 
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
axes[0].imshow(cameraman_resized, cmap='gray') 
axes[0].set_title("Cameraman (256x256)") 
axes[0].axis('off') 

axes[1].imshow(moon_resized, cmap='gray') 
axes[1].set_title("Moon (256x256)") 
axes[1].axis('off') 

plt.show() 