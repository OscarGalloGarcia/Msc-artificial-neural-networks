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

# Deseado calculado manualmente
d = np.array([1 ,1 ,1 ,0 ,1 ,0 ,0 ,0])

#Clasificacion c0, c1
c0 = []
c1 = []

for i, valor in enumerate(X):
    if d[i] == 1:
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
ax.set_title('Clasificación de f(x1,x2,x3)')
ax.legend()
ax.view_init(elev=20, azim=35)
plt.show()


#Perceptron
#valores iniciales
n_patrones, n_entradas = X.shape
w = np.random.rand(3) #Pesos
b = np.random.rand() #Bias
eta = 0.5 #Learning Rate 


#Prediccion
y = np.zeros(n_patrones)
errores = 0 #numero de exitos
epocas = 100
for epoca in range(epocas):
    print(f"epoca = {epoca}") #epoca
    errores = 0
    for i in range(n_patrones):
        #print(f"i = {i}") #Iteracion en el dato
        v = np.dot(w, X[i,:])+b # u = x^T w; v = u + b
        if v >= 0: # Funcion de activacion "Escalon"
            y[i] = 1
        else:
            y[i]=0
        if y[i] != d[i]:
            #Actualizacion de peso y bias
            w = w + eta * (d[i] - y[i]) * X[i,:]
            b = b + eta * (d[i] - y[i])
            errores += 1

    print(f"errores: {errores}")
    print(f"w = {w}")
    print(f"b = {b}")

    if errores == 0: # Si le diste una pasada a los patrones y error quedo 0 entonces tienes los pesos y bias correctos
        break

#Resultado
rango = np.linspace(-0.2, 1.2, 20)
X1, X2 = np.meshgrid(rango, rango)
# Despeje de x3 a partir de w1*x1 + w2*x2 + w3*x3 + b = 0
X3 = -(w[0]*X1 + w[1]*X2 + b) / w[2]

figure = plt.figure(figsize=(9,8))
ax = figure.add_subplot(111, projection='3d')

for p in X:
    for q in X:
        if np.sum(np.abs(p - q)) == 1:   
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                    color='gray', linewidth=0.7, alpha=0.5)

ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2],
           marker='o', s=140, facecolors='none', edgecolors='tab:blue',
           linewidths=2, depthshade=False, label='C0  (f = 0)')

ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2],
           marker='x', s=140, color='tab:red',
           linewidths=2, depthshade=False, label='C1  (f = 1)')

ax.plot_surface(X1, X2, X3, alpha=0.35, color='tab:green', edgecolor='none')

ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2); ax.set_zlim(-0.2, 1.2)
ax.set_title('Plano de decisión aprendido por el perceptrón')
ax.legend(loc='upper left')
ax.view_init(elev=25, azim=60)
plt.tight_layout()
plt.show()

# Verificacion 
for i in range(n_patrones):
    v = np.dot(w, X[i,:]) + b
    print(X[i], "v =", round(v,3), " y =", int(v>=0), " d =", d[i])