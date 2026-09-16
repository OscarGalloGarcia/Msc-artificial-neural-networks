#Oscar Alberto Gallo Gacría 
#Maestría en ciencias en Robótica e Inteligencia Artificial
#Parte 2 
#Clasificación de cuatro clases con dos perceptrones


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
