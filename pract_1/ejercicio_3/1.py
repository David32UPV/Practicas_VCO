from skimage import data 
import matplotlib.pyplot as plt 

# Cargar imágenes de ejemplo 
cameraman = data.camera() # imagen equivalente a cameraman.tif 
moon = data.moon() # imagen equivalente a moon.tif 

# Mostrar 
fig, axes = plt.subplots(1, 2, figsize=(10,5)) 
axes[0].imshow(cameraman, cmap='gray') 
axes[0].set_title("Cameraman") 
axes[0].axis('off') 

axes[1].imshow(moon, cmap='gray') 
axes[1].set_title("Moon") 
axes[1].axis('off') 

plt.show() 