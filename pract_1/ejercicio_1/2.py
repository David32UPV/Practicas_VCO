from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from matplotlib.colors import ListedColormap, BoundaryNorm 
import numpy as np 
from mpl_toolkits.axes_grid1 import make_axes_locatable 

# ---- Ruta de la imagen ---- 
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "paisaje_2.tiff"
im = Image.open(image_path) 
print("Formato, tamaño, modo:", im.format, im.size, im.mode) 

# ---- Convertir la imagen a modo 'P' (paletizada) ---- 
# Puedes cambiar colors=256 a otro número si quieres una paleta inicial distinta 

n_initial_palette = 256 
im_p = im.convert('P', palette=Image.ADAPTIVE, colors=n_initial_palette) 
print("Modo tras conversión a 'P':", im_p.mode) 

# ---- Array de índices (2D): cada píxel es un índice en la paleta ---- 
arr_p = np.array(im_p) # 2D: H x W, valores 0..(palette_size-1) 

# ---- Obtener la paleta completa desde Pillow y convertir a Nx3 ---- 
pal = im_p.getpalette() # lista plana [R,G,B, R,G,B, ...] 
pal_arr = np.array(pal, dtype=np.uint8).reshape(-1, 3) # shape (palette_entries, 3) 
n_palette_entries = pal_arr.shape[0] 
print("Entradas en la paleta:", n_palette_entries) 

# ---- Detectar índices realmente usados en la imagen paletizada ---- 
used_indices = np.unique(arr_p) 
n_used = used_indices.size 
print("Índices usados en la imagen paletizada:", n_used) 

# ---- Construir un ListedColormap a partir de la paleta completa ---- 
cmap = ListedColormap(pal_arr / 255.0) # normalizamos a 0..1 para matplotlib 
norm = BoundaryNorm(np.arange(n_palette_entries + 1) - 0.5, n_palette_entries) 

# ---- Mostrar: imagen paletizada (por índices) y colorbar basada en la paleta ---- 
fig, ax = plt.subplots(figsize=(10, 6)) 

# Mostramos la imagen (2D índices) con el cmap que viene de la paleta 
ax.imshow(arr_p, cmap=cmap, norm=norm) 
ax.axis('off') 
ax.set_title("Imagen (paletizada): visualización por índices") 

# Colocar colorbar a la derecha (ajustado con make_axes_locatable) 
divider = make_axes_locatable(ax) 
cax = divider.append_axes("right", size="5%", pad=0.05) 

# Creamos un mappable para la colorbar usando la misma cmap/norm 
mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm) 
mappable.set_array([]) 

# Ponemos ticks sólo en los índices realmente usados para no saturar la barra. 
# Si hay muchos índices usados, limitamos etiquetas para que no se amontonen. 
if n_used <= 60: 
    ticks = used_indices.tolist() 

else:   
    # mostramos una selección de índices equidistantes (hasta 60 etiquetas) 
    step = max(1, n_used // 60) 
    ticks = used_indices[::step].tolist() 

cbar = fig.colorbar(mappable, cax=cax, boundaries=np.arange(n_palette_entries+1), ticks=ticks) 
cbar.set_label("Índices de la paleta (colores)") 

plt.tight_layout() 
plt.show() 