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

# Convertir de BGR a HSV
nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)

fig = plt.figure(figsize=(14, 6))

# Primera fila: RGB
fig.add_subplot(2, 2, 1, title="Original (RGB)")
plt.imshow(nemo_rgb)

# Plotting the image on 3D plot RGB
r, g, b = cv2.split(nemo_rgb)  # Dividir la imagen RGB convertida
axis1 = fig.add_subplot(2, 2, 2, projection="3d")
pixel_colors_rgb = nemo_rgb.reshape((np.shape(nemo_rgb)[0] * np.shape(nemo_rgb)[1], 3))
norm_rgb = colors.Normalize(vmin=-1.0, vmax=1.0)
norm_rgb.autoscale(pixel_colors_rgb)
pixel_colors_rgb = norm_rgb(pixel_colors_rgb).tolist()
axis1.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors_rgb, marker=".")
axis1.set_xlabel("Red")
axis1.set_ylabel("Green")
axis1.set_zlabel("Blue")
axis1.set_title("RGB 3D Color Space")

# Segunda fila: HSV
fig.add_subplot(2, 2, 3, title="Original (HSV)")
plt.imshow(cv2.cvtColor(nemo_hsv, cv2.COLOR_HSV2RGB))

# Plotting the image on 3D plot HSV
h, s, v = cv2.split(nemo_hsv)  # Dividir la imagen HSV
axis2 = fig.add_subplot(2, 2, 4, projection="3d")
pixel_colors_hsv = nemo_hsv.reshape((np.shape(nemo_hsv)[0] * np.shape(nemo_hsv)[1], 3))

# Normalizar HSV para visualización (OpenCV usa H: 0-180, S: 0-255, V: 0-255)
pixel_colors_hsv_norm = pixel_colors_hsv.copy().astype(np.float32)
pixel_colors_hsv_norm[:, 0] = pixel_colors_hsv_norm[:, 0] / 180.0  # H: 0-1
pixel_colors_hsv_norm[:, 1] = pixel_colors_hsv_norm[:, 1] / 255.0  # S: 0-1
pixel_colors_hsv_norm[:, 2] = pixel_colors_hsv_norm[:, 2] / 255.0  # V: 0-1

norm_hsv = colors.Normalize(vmin=-1.0, vmax=1.0)
norm_hsv.autoscale(pixel_colors_hsv_norm)
pixel_colors_hsv_norm = norm_hsv(pixel_colors_hsv_norm).tolist()

axis2.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors_hsv_norm, marker=".")
axis2.set_xlabel("Hue")
axis2.set_ylabel("Saturation")
axis2.set_zlabel("Value")
axis2.set_title("HSV 3D Color Space")

plt.tight_layout()
plt.show()

# ============ SEGMENTACIÓN DE COLOR NARANJA ============
# Definir rango de color naranja en HSV
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

# Segmentar Nemo usando inRange() function
mask_orange = cv2.inRange(nemo_hsv, light_orange, dark_orange)

# Bitwise-AND mask y imagen original
result = cv2.bitwise_and(nemo, nemo, mask=mask_orange)

# Convertir resultado a RGB para mostrar correctamente
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Mostrar mascara y resultado
fig2, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].imshow(mask_orange, cmap="gray")
axs[0].set_title("Máscara de Color Naranja")
axs[0].axis('off')

axs[1].imshow(result_rgb)
axs[1].set_title("Nemo Segmentado (Solo Naranja)")
axs[1].axis('off')

plt.tight_layout()
plt.show()
