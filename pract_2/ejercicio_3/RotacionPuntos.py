import matplotlib.pyplot as plt
import numpy as np
import string
import math

a, b, c, d = (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1) # points a, b, c y d en coordenadas homogéneas
A = np.array([a, b, c, d])
T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) # escalado 2x
T_r = np.array ([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # rotación 90 grados
T = T_s @ T_r

# 3x3 Identity transformation matrix
I = np.eye(3)
color_lut = 'rgbc'
fig = plt.figure()
ax = plt.gca()
xs_s = []
ys_s = []

i=0
for row in A:
    output_row = T @ row # transformación de puntos
    x, y, h = row
    x_s, y_s, k = output_row
    xs_s.append(x_s)
    ys_s.append(y_s)
    c = color_lut[i]
    plt.scatter(x, y, color=c)
    plt.scatter(x_s, y_s, color=c)
    plt.text(x + 0.15, y, f"{string.ascii_letters[int(i)]}")
    plt.text(x_s + 0.15, y_s, f"{string.ascii_letters[int(i)]}'")
    i+=1
    
xs_s.append(xs_s[0])
ys_s.append(ys_s[0])
plt.plot(x_s, y_s, color="gray", linestyle='dotted')
plt.plot(xs_s, ys_s, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-10, 10, 1.0))
ax.set_yticks(np.arange(-10, 10, 1.0))
plt.grid()
plt.show()