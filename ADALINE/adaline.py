#Oscar Alberto Gallo García
#Maestría en Ciencias en Robótica e Inteligencia Artificial
#ADALINE (Widrow-Hoff / regla delta)

import numpy as np

class adaline:
    def __init__(self, n_inputs, eta, paro=1e-6, shuffle=True, random_state=1):
        self.rgen = np.random.RandomState(random_state) # Semilla para tener datos pequeños de inicio
        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=n_inputs) #Pesos
        self.b = self.rgen.normal(loc=0.0, scale=0.01) #Bias
       
        self.eta = eta #Learning Rate
        self.paro = paro #Criterio de paro (Cambio minimo del MSE)
        self.mse = [] #MSE por epoca
        self.shuffle = shuffle # Mezcla patrones al inicio

    def net_input(self, X):
        return np.dot(X, self.w) + self.b # v = X^T w + b

    def activacion(self, v):
        return v # Funcion de activacion "Identidad" (salida lineal)

    def entrenamiento(self, X, d, epocas=50):
        n_patrones  = X.shape[0]
        self.mse = []

        for epoca in range(epocas):
            indices = np.arange(n_patrones)
            if self.shuffle: #Mezcla los patrones
                self.rgen.shuffle(indices)

            # Stochastic Gradient Descent (SDG)
            for k in indices:
                X_k = X[k]
                v = np.dot(X_k, self.w) + self.b # v = X^T w + b
                y = self.activacion(v=v)
                e = d[k] - y # Error

                #Actualizacion de pesos
                self.w = self.w + self.eta * e * X_k
                #Actualizacion de Bias
                self.b = self.b + self.eta * e

            #MSE
            e_epoca = d - self.activacion(self.net_input(X))
            self.mse.append(np.mean(e_epoca ** 2))

            #Criterio de paro
            if not np.isfinite(self.mse[-1]):
                break
            if epoca > 0 and abs(self.mse[-2] - self.mse[-1]) < self.paro:
                break 
        return self

    def prediccion(self, X):
        return np.where(self.activacion(self.net_input(X)) >= 0, 1, -1) 

    def verificacion(self, X, d):
        n_patrones, n_entradas = X.shape
        for i in range(n_patrones):
            v = self.net_input(X[i,:])
            print(X[i], "v =", round(v,3), " y =", self.prediccion(X[i,:]), " d =", d[i])


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X_a = rng.normal(loc=[-1.5, -1.5], scale=0.7, size=(100, 2))
    X_b = rng.normal(loc=[1.5, 1.5], scale=0.7, size=(100, 2))
    X = np.vstack([X_a, X_b])
    d = np.hstack([-np.ones(100), np.ones(100)])
 
    for eta in [0.001, 0.01, 0.1]:
        a = adaline(n_inputs=2, eta=eta)
        a.entrenamiento(X, d, epocas=50)
        acc = np.mean(a.prediccion(X) == d)
        print(f"eta = {eta:<6} | MSE inicial {a.mse[0]:.4f} "
              f"| MSE final {a.mse[-1]:.4f} "
              f"| epocas {len(a.mse):>3} "
              f"| accuracy {acc:.3f}")