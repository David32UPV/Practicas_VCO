# Python program to print nXn
# checkerboard pattern using numpy
 
import matplotlib.pyplot as plt
import numpy as np  
import string
from PIL import Image
import math


# function to return a Checkerboard graylevel image of n x n squares each of size 'sz'
def checkboard(grid, square_size):
     
    print("Checkerboard pattern:")
 
    # create a n * n matrix
    x = np.zeros((grid, grid), dtype = np.uint8)
 
    # fill with 255 the alternate rows and columns
    x[1::2, ::2] = 255
    x[::2, 1::2] = 255
    
    sz = grid * square_size 
    size = (sz,sz)
    img = Image.fromarray(x)
    img_res = img.resize(size, resample=Image.Resampling.NEAREST)
    return img_res

# Main code

img =checkboard(10,20)   
plt.figure(figsize=(9,5))

# Lógica de matrices
tx, ty = 100, 50
angulo = 30
sen = math.sin(math.radians(angulo))
cos = math.cos(math.radians(angulo))

T_traslacion = np.array([
[1, 0, tx],
[0, 1, ty],
[0, 0, 1]])

# rotate - angle 
T_rotate = np.array([ 
[cos, -sen, 0], 
[sen, cos, 0], 
[0, 0, 1]])

# center original to image center
T_center = np.array([
[1, 0, -img.width/2],
[0, 1, -img.height/2],
[0, 0, 1]])

T_uncenter = np.array([
[1, 0, img.width/2],
[0, 1, img.height/2],   
[0, 0, 1]])

T_total = T_traslacion @ T_uncenter @ T_rotate @ T_center
T_inv = np.linalg.inv(T_total) # Image.transform uses the inverse transform matrix 
img_transformed = img.transform((400, 400), Image.Transform.AFFINE, data=T_inv.flatten()[:6] ,resample=Image.BICUBIC) 

plt.close('all')  # cierra figuras previas por si acaso

plt.figure(figsize=(12, 8)) 
plt.subplot(1, 2, 1) 
plt.title("Imagen original") 
plt.imshow(np.asarray(img)) 
plt.axis("off") 

plt.subplot(1, 2, 2) 
plt.title("Imagen transformada") 
plt.imshow(np.asarray(img_transformed)) 
plt.axis("off") 

plt.tight_layout() 

plt.show() 

img.save('checkerboard.jpg')