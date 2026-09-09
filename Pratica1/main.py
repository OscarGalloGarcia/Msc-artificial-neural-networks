#Oscar Alberto Gallo García
#Maestría en Ciencias en Robótica e Inteligencia Artificial

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1, 1],
    [1 ,1, 0],
    [1 ,0 ,1],
    [1 ,0 ,0],
    [0 ,1 ,1],
    [0 ,1 ,0],
    [0 ,0 ,1],
    [0 ,0 ,0],
])
#print(X.shape)

f = np.array([1 ,1 ,1 ,0 ,1 ,0 ,0 ,0])

#Clasificacion c0, c1
c0 = []
c1 = []

for i, valor in enumerate(X):
    if f[i] == 1:
        c1.append(valor)
    else:
        c0.append(valor)

c0 = np.array(c0)
c1 = np.array(c1)

print(f"Clase 0: \n{c0}")
print(f"Clase 1: \n{c1}")

#Grafica
figure = plt.figure(figsize=(8,7))
ax = figure.add_subplot(111, projection='3d')

ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2],
           marker='o', s=120, facecolors='none', edgecolors='tab:blue',
           label='C0  (f = 0)')

ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2],
           marker='x', s=120, color='tab:red',
           label='C1  (f = 1)')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Patrones de f(x1,x2,x3) por clase')
ax.legend()
ax.view_init(elev=20, azim=35)
plt.show()