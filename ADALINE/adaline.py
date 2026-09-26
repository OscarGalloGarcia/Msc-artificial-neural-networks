
import numpy as np

class adaline(object):
    
    def __init__(self, eta, epocas, random_state=1):
        self.eta = eta
        self.epocas = epocas
        self.random_state = random_state

    def fit(self, X, y):

        #Valores aleatorios pequeños con semilla 0.01
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0,scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epocas):
            n_input = self.n_imput(X)
            #Salida
            y_o = self.activation(n_input)
            error = (y - y_o)

            self.w[1:] += self.eta * X.T.dot(error) #pesos
            self.w[0] += self.eta * error.sum() #bias
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)

        return self