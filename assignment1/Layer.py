import numpy as np

class Layer:
    def __init__(self, input_size, output_size, learning_rate):
        self.whigts = np.random.uniform(size=(input_size, output_size), low=-1.0, high=1.0)
        self.bias = np.random.uniform(size=(1, output_size), low=-1.0, high=1.0)
        self.input = None
        self.output = None
        self.learning_rate = learning_rate

    def forward_prop(self, inputs):
        self.input = inputs
        self.output = np.dot(self.input, self.whigts) + self.bias
        return self.output

    def backward_prop(self, output_err):
        input_err = np.dot(output_err, self.whigts.T)
        weights_err = np.dot(self.input.T, output_err)

        self.whigts -= self.learning_rate * weights_err
        self.bias -= self.learning_rate * output_err
        return input_err

