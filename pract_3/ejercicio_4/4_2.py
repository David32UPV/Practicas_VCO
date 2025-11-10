import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
src = np.array(Image.open(image_path).convert('RGB'))

# Definición de kernels (float32)
kernel_emboss1 = np.array([[0, -1, -1],
                           [1,  0, -1],
                           [1,  1,  0]], dtype=np.float32)

kernel_blur = np.ones((3,3), dtype=np.float32) / 9.0

kernel_emboss2 = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]], dtype=np.float32)

kernel_edge = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)

kernel_sharp = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float32)

# Función auxiliar que aplica filter2D mostrando uso de parámetros:
def apply_kernel(img, kernel, ddepth=-1, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT, use_float_convert=False):
    """
    Aplica cv2.filter2D con los parámetros indicados y devuelve uint8 listo para mostrar.
    - ddepth: profundidad de destino (ej: -1 mantiene la del src, cv2.CV_32F para precisión)
    - anchor: punto origen del kernel, (-1,-1) -> centro
    - delta: valor añadido al resultado (útil para emboss para desplazar a rango visible)
    - borderType: comportamiento en bordes (BORDER_DEFAULT, BORDER_REPLICATE, BORDER_REFLECT, ...)
    - use_float_convert: si True, realiza la convolución en CV_32F y convierte con convertScaleAbs
    """
    if use_float_convert:
        dst_f = cv2.filter2D(img, cv2.CV_32F, kernel, anchor=anchor, delta=delta, borderType=borderType)
        dst = cv2.convertScaleAbs(dst_f)  # convierte y normaliza para display uint8
    else:
        dst = cv2.filter2D(img, ddepth, kernel, anchor=anchor, delta=delta, borderType=borderType)
        # filter2D ya devuelve uint8 si ddepth=-1 y src es uint8
    return dst

# Aplicaciones prácticas mostrando distintos parámetros vistos en la teoría:
# - ddepth = -1 (mantiene uint8)
# - anchor = (-1,-1) (centro del kernel)
# - delta: para emboss puede ser útil sumar 128 para centrar resultados
# - borderType: probar reflect / replicate para evitar artefactos en bordes

dst_emboss1 = apply_kernel(src, kernel_emboss1, ddepth=-1, anchor=(-1,-1), delta=128, borderType=cv2.BORDER_REPLICATE)
dst_blur    = apply_kernel(src, kernel_blur, ddepth=-1, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT)
dst_emboss2 = apply_kernel(src, kernel_emboss2, ddepth=-1, anchor=(-1,-1), delta=128, borderType=cv2.BORDER_REPLICATE)

# Edge detection: usar CV_32F + convertScaleAbs para capturar correctamente valores negativos
dst_edge = apply_kernel(src, kernel_edge, use_float_convert=True, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_REFLECT)

dst_sharp   = apply_kernel(src, kernel_sharp, ddepth=-1, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT)

# Sepia (transformación de color, no es convolución espacial)
sepia_kernel = np.array([[0.272, 0.534, 0.1311],
                         [0.349, 0.686, 0.168 ],
                         [0.393, 0.769, 0.189 ]], dtype=np.float32)
dst_sepia = cv2.transform(src, sepia_kernel)
dst_sepia = np.clip(dst_sepia, 0, 255).astype(np.uint8)

# Listado de imágenes y títulos para mostrar en la misma ventana
images = [
    (src, "Original"),
    (dst_emboss1, "Realce (Emboss 1)\n(delta=128, border=REPLICATE)"),
    (dst_blur, "Suavizado (Blur avg)"),
    (dst_emboss2, "Realce 2 (Emboss 2)\n(delta=128, border=REPLICATE)"),
    (dst_edge, "Paso alto (Edge)\n(CV_32F -> convertScaleAbs, border=REFLECT)"),
    (dst_sharp, "Agudizado (Sharpen)"),
    (dst_sepia, "Sepia")
]

# Mostrar todo en la misma ventana: 2 filas x 4 columnas (último hueco vacío)
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.ravel()

for i, (img, title) in enumerate(images):
    axs[i].imshow(img)
    axs[i].set_title(title, fontsize=10)
    axs[i].axis('off')

# Último subplot vacío
axs[len(images)].axis('off')

plt.tight_layout()
plt.show()