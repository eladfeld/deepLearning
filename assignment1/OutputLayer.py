import numpy as np
from Layer import Layer

def softmax(z):
    m = np.max(z, axis=1, keepdims=True)
    ZminusM  = z - m
    e_x = np.exp(ZminusM)
    sum_ = np.sum(e_x, axis=1, keepdims=True)
    output = e_x / sum_
    return output

def softmax_grad_matrix(z):
    sm = softmax(z)
    n = z.shape[0]
    val = 0
    output = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            val = (sm[i] * (1 - sm[i])) if (i == j) else (- sm[i] * sm[j])
            output[i][j] = val
    return output



class OutputLayer(Layer):
    def __init__(self, input_size, output_size, learning_rate):
        super().__init__(input_size, output_size, learning_rate)
        self.activation = softmax

    def backward_prop(self, reals):
        output_err = self.output - reals
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)

        self.weights -= self.learning_rate * weights_err
        self.bias -= self.learning_rate * np.mean(output_err)
        return input_err
