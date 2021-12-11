import numpy as np
from Layer import Layer


class ResLayer(Layer):

    def __init__(self, input_size, output_size, learning_rate):
        super().__init__(input_size, output_size, learning_rate)
        self.weights_2 = np.random.uniform(size=(input_size, input_size), low=-1.0, high=1.0)

    def forward_prop(self, inputs):
        self.input = np.atleast_2d(inputs)
        self.z = np.dot(self.input, self.weights) + self.bias
        output = self.activation(self.z)
        output = output.dot(self.weights_2)
        self.output = output + inputs
        return self.output

    def backward_prop(self, output_err):
        W1 = self.weights
        W2 = self.weights_2
        X = self.input
        V = output_err
        Z = self.z
        VdotW2T = V @ W2.T
        activationPrimeZ = self.activation_prime(Z)
        dW1 = X.T @ (activationPrimeZ * VdotW2T)
        dW2 = V.T @ activationPrimeZ
        dB = activationPrimeZ * VdotW2T
        input_error = V + (activationPrimeZ * VdotW2T) @ W1

        self.weights -= self.learning_rate * dW1
        self.weights_2 -= self.learning_rate * dW2
        self.bias -= self.learning_rate * np.mean(dB)

        return input_error





