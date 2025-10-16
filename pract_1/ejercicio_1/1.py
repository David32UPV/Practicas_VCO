from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import numpy as np 

im = Image.open("Paisaje_de_Albacete.jpg")

print(im.format, im.size, im.mode) 

img = np.asarray(im) 
imgplot = plt.imshow(img) 
plt.show() 