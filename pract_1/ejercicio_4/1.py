from skimage import data 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # necesario para proyección 3D en algunas versiones 

# Cargar la imagen (grayscale) 
cameraman = data.camera().astype(float) # shape (H, W) 

# Parámetro: downsample para acelerar la representación 
ds = 4 # 1 = sin muestreo; prueba 2, 4... según tu máquina 

h, w = cameraman.shape 
x = np.arange(0, w) 
y = np.arange(0, h) 
X, Y = np.meshgrid(x, y) # numpy.meshgrid 

# Submuestreo (si ds>1) 
Xs = X[::ds, ::ds] 
Ys = Y[::ds, ::ds] 
Zs = cameraman[::ds, ::ds] 

# ==== Figura 3D usando contourf ==== 
fig = plt.figure(figsize=(10,8)) 
ax = fig.add_subplot(111, projection='3d') 

# Usamos ax.contourf (equivalente a plt.contourf pero en ejes 3D) 
# contourf en 3D dibuja "capas" rellenadas a varios niveles de z, 
# con esto conseguimos una apariencia de superficie escalonada. 

levels = 60 
cset = ax.contourf(Xs, Ys, Zs, levels=levels, cmap='viridis', zsort='average') 

# Opciones visuales 
ax.set_xlabel('X (pix)') 
ax.set_ylabel('Y (pix)') 
ax.set_zlabel('Intensidad') 
ax.set_title('Cameraman representado como superficie (contourf en 3D)') 
ax.view_init(elev=35, azim=-60) 

# Barra de color 
fig.colorbar(cset, ax=ax, shrink=0.6, aspect=12, label='Intensidad') 

plt.tight_layout() 

plt.show() 

 