import matplotlib.pyplot as plt 
import numpy as np 
import string 
from PIL import Image 
import math 

img = Image.open('Paisaje_de_Albacete.jpg') 
img.size # (640, 480) 

angle = 10 
sen = math.sin(math.radians(angle)) 
cos = math.cos(math.radians(angle)) 

T_pos1000 = np.array([
[1, 0, 1000],
[0, 1, 1000],
[0, 0, 1]])

# rotate - angle 
T_rotate = np.array([ 
[cos, -sen, 0], 
[sen, cos, 0], 
[0, 0, 1]]) 

# scale
T_scale = np.array([
[3, 0, 0],
[0, 3, 0],
[0, 0, 1]])

# center original to image center
T_center = np.array([
[1, 0, -img.width/2],
[0, 1, -img.height/2],
[0, 0, 1]])

T = T_pos1000 @ T_rotate @ T_scale @ T_center
T_inv = np.linalg.inv(T) # Image.transform uses the inverse transform matrix 
img_transformed = img.transform((2000, 2000), Image.Transform.AFFINE, data=T_inv.flatten()[:6] ,resample=Image.NEAREST) 

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

plt.show() 