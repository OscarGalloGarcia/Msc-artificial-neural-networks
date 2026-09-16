#Oscar Alberto Gallo Gacría 
#Maestría en ciencias en Robótica e Inteligencia Artificial
#Parte 2 
#Clasificación de cuatro clases con dos perceptrones

import numpy as np
import matplotlib.pyplot as plt

#Entradas (x1, x2)
X = np.array([
    [-2.0,  0.0],
    [-1.0,  0.5],
    [-1.5, -0.2],
    [ 0.0,  0.5],
    [ 0.0, -1.0],
    [ 0.5, -0.5],
    [-0.5, -1.0],
    [ 0.3,  0.1],
    [ 0.0,  2.0],
    [ 0.5,  1.5],
    [-0.5,  2.0],
    [ 1.0,  1.5],
    [ 2.0,  0.0],
    [ 1.5,  0.5],
    [ 2.0, -0.5],
    [ 1.5,  1.0],
])
#print(X.shape)

#Clase de cada patron
clases = np.array(['C1','C1','C1','C1',
                    'C2','C2','C2','C2',
                    'C3','C3','C3','C3',
                    'C4','C4','C4','C4'])

# Deseado por neurona (codificacion de clases)
# C1 = 0 0 | C2 = 0 1 | C3 = 1 0 | C4 = 1 1
d1 = np.array([0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1])
d2 = np.array([0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1])

#Separacion por clase para graficar
c1 = X[clases == 'C1']
c2 = X[clases == 'C2']
c3 = X[clases == 'C3']
c4 = X[clases == 'C4']

#Grafica del conjunto de entrenamiento
figure = plt.figure(figsize=(7,7))

plt.scatter(c1[:,0], c1[:,1], marker='o', s=120, facecolors='none', edgecolors='tab:blue', label='C1 (0,0)')
plt.scatter(c2[:,0], c2[:,1], marker='s', s=120, facecolors='none', edgecolors='tab:orange', label='C2 (0,1)')
plt.scatter(c3[:,0], c3[:,1], marker='^', s=120, facecolors='none', edgecolors='tab:green', label='C3 (1,0)')
plt.scatter(c4[:,0], c4[:,1], marker='x', s=120, color='tab:red', label='C4 (1,1)')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Conjunto de entrenamiento')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

class perceptron:
    def __init__(self, n_inputs, eta):
        self.w = np.random.rand(n_inputs) #Pesos
        self.b = np.random.rand() #bias
        self.eta = eta

    def entrenamiento(self, X, d, epocas=100):
        n_patrones, n_entradas = X.shape
        y = np.zeros(n_patrones)

        for epoca in range(epocas):
            print(f"epoca = {epoca}") #epoca
            errores = 0
            for i in range(n_patrones):
                v = np.dot(self.w, X[i,:])+self.b # u = x^T self.w; v = u + b
                if v >= 0: # Funcion de activacion
                    y[i] = 1
                else:
                    y[i]=0
                if y[i] != d[i]:
                    #Actualizacion de peso y bias
                    self.w = self.w + self.eta * (d[i] - y[i]) * X[i,:]
                    self.b = self.b + self.eta * (d[i] - y[i])
                    errores += 1

            print(f"errores: {errores}")
            print(f"w = {self.w}")
            print(f"b = {self.b}")

            if errores == 0: # Si le diste una pasada a los patrones y error quedo 0 entonces tienes los pesos y bias correctos
                break

    def verificacion(self, X, d):
        n_patrones, n_entradas = X.shape
        #Clasificacion de confirmacion:
        for i in range(n_patrones):
            v = np.dot(self.w, X[i,:]) + self.b
            print(X[i], "v =", round(v,3), " y =", int(v>=0), " d =", d[i])


#Entrenamiento y verificacion del perceptron 1 (neurona d1)
p1 = perceptron(n_inputs=2, eta=0.1)
p1.entrenamiento(X, d1)
p1.verificacion(X, d1)

#Entrenamiento y verificacion del perceptron 2 (neurona d2)
p2 = perceptron(n_inputs=2, eta=0.1)
p2.entrenamiento(X, d2)
p2.verificacion(X, d2)

