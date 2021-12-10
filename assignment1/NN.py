import numpy as np
from Layer import Layer

class NN:
    def __init__(self, loss, d_loss, layer_dims, learning_rate):
        self.layers = []
        self.loss = loss
        self.d_loss = d_loss
        for i in range(len(layer_dims) - 1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], learning_rate))


    def add(self, layer):
        self.layers.append(layer)


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

                err += self.loss(y_train[j], output)

                error = self.d_loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error)
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
