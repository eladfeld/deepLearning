import numpy as np
from Layer import Layer
from ActiveLayer import ActiveLayer



def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;



class NN:
    def __init__(self, loss, d_loss, learning_rate,  architecture):
        self.layers = []
        self.loss = loss
        self.d_loss = d_loss
        for i in range(len(architecture) - 1):
            self.layers.append(Layer(architecture[i], architecture[i + 1], learning_rate))
            self.layers.append(ActiveLayer(tanh, tanh_prime))

    def predict(self, input):
        samples = len(input)
        result = []
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                err +=self.loss(y_train[j], output)

                error = self.d_loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error)
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
