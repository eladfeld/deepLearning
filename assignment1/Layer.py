import numpy as np


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


class Layer:
    def __init__(self, input_size, output_size, learning_rate):
        self.whigts = np.random.uniform(size=(input_size, output_size), low=-1.0, high=1.0)
        self.bias = np.random.uniform(size=(1, output_size), low=-1.0, high=1.0)
        self.input = None
        self.output = None
        self.z = None
        self.learning_rate = learning_rate
        self.activation = tanh
        self.activation_prime = tanh_prime

    def forward_prop(self, inputs):
        self.input = np.atleast_2d(inputs)
        self.z = np.dot(self.input, self.whigts) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward_prop(self, output_err):
        output_err = self.activation_prime(self.z) * output_err
        input_err = np.dot(output_err, self.whigts.T)
        weights_err = np.dot(self.input.T, output_err)

        self.whigts -= self.learning_rate * weights_err
        self.bias -= self.learning_rate * output_err
        return input_err

