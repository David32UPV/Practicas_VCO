import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "b1.jpg"
img = np.asarray(Image.open(image_path).convert('L'))  # para escala de grises

def imadjustLUT(img, a, b, mode='band'):
    """
    img : numpy uint8 image (grayscale)
    a, b : int in [0,255]
    mode : 'band' or 'peak'
      - 'band'  : fuera de [a,b] -> 0, dentro -> 255
      - 'peak'  : identidad lineal salvo que en [a,b] se fuerza a 255;
                  después de b se retoma la identidad (v=u)
    """
    a = int(np.clip(a, 0, 255))
    b = int(np.clip(b, 0, 255))
    if a > b:
        a, b = b, a

    if mode == 'band':
        LUT = np.zeros(256, dtype=np.uint8)
        LUT[a:b+1] = 255
    elif mode == 'peak':
        LUT = np.arange(256, dtype=np.uint8)
        LUT[a:b+1] = 255
    else:
        raise ValueError("modo debe ser 'band' o 'peak'")

    img_out = LUT[img]
    return img_out, LUT

# Ejemplo de uso y visualización
if __name__ == "__main__":
    # Band con a=60, b=140 (según petición)
    out_band, lut_band = imadjustLUT(img, a=60, b=140, mode='band')

    # Peak (lineal con plateau en [a,b]) - mantiene la recta identidad fuera del intervalo
    out_peak, lut_peak = imadjustLUT(img, a=60, b=140, mode='peak')

    # Mostrar solo: imagen original, resultado (band), LUT band; y original, resultado (peak), LUT peak
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0,0].imshow(img, cmap='gray'); axs[0,0].set_title('Original'); axs[0,0].axis('off')
    axs[0,1].imshow(out_band, cmap='gray'); axs[0,1].set_title('Band (a=60,b=140)'); axs[0,1].axis('off')
    axs[0,2].plot(lut_band); axs[0,2].set_title('LUT Band'); axs[0,2].set_xlim(0,255)

    axs[1,0].imshow(img, cmap='gray'); axs[1,0].set_title('Original'); axs[1,0].axis('off')
    axs[1,1].imshow(out_peak, cmap='gray'); axs[1,1].set_title('Peak (a=60,b=140)'); axs[1,1].axis('off')
    axs[1,2].plot(lut_peak); axs[1,2].set_title('LUT Peak'); axs[1,2].set_xlim(0,255)

    plt.tight_layout()
    plt.show()

