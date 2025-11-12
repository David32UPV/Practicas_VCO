import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
import cv2
from pathlib import Path 

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "nemo0.jpg"
img = np.asarray(Image.open(image_path))

nemo = cv2.imread(str(image_path))

# Convertir de BGR a RGB para que Matplotlib lo muestre correctamente
nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

fig = plt.figure()
fig.add_subplot(1, 2, 1, title="Original")
plt.imshow(nemo_rgb)  # Mostrar la imagen convertida a RGB

# Plotting the image on 3D plot
r, g, b = cv2.split(nemo_rgb)  # Dividir la imagen RGB convertida
axis = fig.add_subplot(1, 2, 2, projection="3d")
pixel_colors = nemo_rgb.reshape((np.shape(nemo_rgb)[0] * np.shape(nemo_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.0, vmax=1.0)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

#OpenCV lee las imágenes en formato BGR (Blue-Green-Red), mientras que Matplotlib espera RGB (Red-Green-Blue). Cuando usas cv2.imread(), los canales están en orden BGR, pero al mostrarlos con plt.imshow() sin convertir, Matplotlib los interpreta como RGB, intercambiando rojo y azul.