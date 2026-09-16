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
