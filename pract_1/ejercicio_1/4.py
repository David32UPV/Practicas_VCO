from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from matplotlib.colors import ListedColormap, BoundaryNorm 
import numpy as np 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "galaxia.jpg"
im = Image.open(image_path) 
print("Formato, tamaño, modo:", im.format, im.size, im.mode) 

# ---- Convertir la imagen a modo 'P' con paleta inicial (256 colores) ---- 
n_initial_palette = 256 
im_p = im.convert('P', palette=Image.ADAPTIVE, colors=n_initial_palette) 
arr_p = np.array(im_p) # índices 2D 
pal_p = np.array(im_p.getpalette(), dtype=np.uint8).reshape(-1,3) 
used_indices_p = np.unique(arr_p) 
n_used_p = used_indices_p.size 

print(f"Imagen paletizada original: {n_used_p} índices usados") 

# ---- Reducir la imagen a 16 colores ---- 
n_target = 16 
im_16 = im.convert('P', palette=Image.ADAPTIVE, colors=n_target) 
arr_16 = np.array(im_16) 
pal_16 = np.array(im_16.getpalette()[:n_target*3], dtype=np.uint8).reshape(-1,3) 
used_indices_16 = np.unique(arr_16) 
n_used_16 = used_indices_16.size 
print(f"Imagen reducida a 16 colores: {n_used_16} índices usados") 

# ---- Mostrar lado a lado ---- 
fig, axes = plt.subplots(1, 2, figsize=(14,6)) 

# --- Imagen paletizada original --- 
ax0 = axes[0] 
cmap_p = ListedColormap(pal_p / 255.0) 
norm_p = BoundaryNorm(np.arange(len(pal_p)) - 0.5, len(pal_p)) 
im0 = ax0.imshow(arr_p, cmap=cmap_p, norm=norm_p) 
ax0.set_title(f"Paletizada original ({n_used_p} índices usados)") 
ax0.axis('off') 

divider0 = make_axes_locatable(ax0) 
cax0 = divider0.append_axes("right", size="5%", pad=0.05) 
mappable0 = mpl.cm.ScalarMappable(cmap=cmap_p, norm=norm_p) 
mappable0.set_array([]) 

# Colorbar con ticks limitados para no saturar 
if n_used_p <= 60: 
    ticks0 = used_indices_p.tolist() 
else: 
    step0 = max(1, n_used_p // 60) 

ticks0 = used_indices_p[::step0].tolist() 
cbar0 = fig.colorbar(mappable0, cax=cax0, boundaries=np.arange(len(pal_p)+1), ticks=ticks0) 
cbar0.set_label("Índices paleta original") 

# --- Imagen reducida a 16 colores --- 
ax1 = axes[1] 
cmap_16 = ListedColormap(pal_16 / 255.0) 
norm_16 = BoundaryNorm(np.arange(n_target+1) - 0.5, n_target) 
im1 = ax1.imshow(arr_16, cmap=cmap_16, norm=norm_16) 
ax1.set_title(f"Reducida a {n_target} colores ({n_used_16} índices usados)") 
ax1.axis('off') 

divider1 = make_axes_locatable(ax1) 
cax1 = divider1.append_axes("right", size="5%", pad=0.05) 
mappable1 = mpl.cm.ScalarMappable(cmap=cmap_16, norm=norm_16) 
mappable1.set_array([]) 

# Colorbar con ticks según índices usados 

if n_used_16 <= 60: 
    ticks1 = used_indices_16.tolist() 

else: 
    step1 = max(1, n_used_16 // 60) 
    ticks1 = used_indices_16[::step1].tolist() 
    cbar1 = fig.colorbar(mappable1, cax=cax1, boundaries=np.arange(n_target+1), ticks=ticks1) 
    cbar1.set_label("Índices paleta 16 colores") 

plt.tight_layout() 

plt.show() 