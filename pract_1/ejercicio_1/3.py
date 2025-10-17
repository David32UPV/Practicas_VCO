from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from matplotlib.colors import ListedColormap, BoundaryNorm 
import numpy as np 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "paisaje_2.tiff"
im = Image.open(image_path) 
print("Formato, tamaño, modo:", im.format, im.size, im.mode) 

# ---- Extraer colores únicos de la imagen original (RGB) ---- 
im_rgb = im.convert("RGB") 
arr_rgb = np.array(im_rgb) # H x W x 3 
pixels = arr_rgb.reshape(-1, 3) 

# obtener únicos (exacto si la imagen no es gigante) 
unique_colors = np.unique(pixels, axis=0) 
n_unique = unique_colors.shape[0] 
print("Colores únicos detectados (original):", n_unique) 

# ---- REDUCIR a 16 colores ---- 
n_target = 16 
im_16 = im_rgb.convert('P', palette=Image.ADAPTIVE, colors=n_target) # <-- aquí pedimos 16 colores 
arr_16 = np.array(im_16) # array 2D de índices 
used_indices = np.unique(arr_16) 
n_used = used_indices.size 

print(f"Colores (índices) usados en la versión a {n_target} colores: {n_used}") 

# ---- Obtener la paleta (RGB) para matplotlib ---- 
pal = im_16.getpalette()[:n_target*3] # tomamos los primeros n_target colores 
pal_arr = np.array(pal, dtype=np.uint8).reshape(-1,3) / 255.0 
cmap = ListedColormap(pal_arr) 
norm = BoundaryNorm(np.arange(n_target + 1) - 0.5, n_target) 

# ---- Mostrar lado a lado: original | reducida (con colorbar) ---- 
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

# Original (izquierda) 
ax0 = axes[0] 
ax0.imshow(arr_rgb) # imagen original RGB 
ax0.set_title(f"Original — ~{n_unique} colores") 
ax0.axis('off') 

# Reducida (derecha) — mostramos índices con la paleta creada 
ax1 = axes[1] 
imshow = ax1.imshow(arr_16, cmap=cmap, norm=norm) 
ax1.set_title(f"Reducida a {n_target} colores — {n_used} índices usados") 
ax1.axis('off') 

# Colorbar al lado de la imagen reducida 
divider = make_axes_locatable(ax1) 
cax = divider.append_axes("right", size="6%", pad=0.05) 
mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm) 
mappable.set_array([]) 

# Ponemos ticks sólo en los índices usados para mayor claridad 
ticks = used_indices.tolist() 
# Si hay muchos índices, limitar etiquetas para que no se amontonen 
if len(ticks) > 40: 
    ticks_display = ticks[::max(1, len(ticks)//40)] 
else: 
    ticks_display = ticks 

cbar = fig.colorbar(mappable, cax=cax, boundaries=np.arange(n_target+1), ticks=ticks_display) 
cbar.set_label("Índices de paleta (colores)") 


plt.tight_layout() 
plt.show() 