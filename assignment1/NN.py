import matplotlib.pyplot as plt
import numpy as np

from Layer import Layer
from OutputLayer import OutputLayer
from SGD import unison_shuffled_copies


def num_correct(pred, real):
    (n, m) = pred.shape
    p = np.argmax(pred, axis=1)
    r = np.argmax(real, axis=1)
    n_correct = np.sum(p == r)
    return n_correct


class NN:
    def __init__(self, loss, d_loss, layer_dims, learning_rate):
        self.layers = []
        self.loss = loss
        self.d_loss = d_loss
        for i in range(len(layer_dims) - 2):
            self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], learning_rate))
        self.layers.append((OutputLayer(layer_dims[-2], layer_dims[-1], learning_rate)))

    def predict(self, input):
        samples = len(input)
        result = []
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, batch_size):
        samples = len(x_train)
        error_list, acc_list = [], []
        for i in range(epochs):
            x_train, y_train = unison_shuffled_copies(x_train, y_train)
            err = 0
            n_correct = 0
            for j in range(int(samples / batch_size)):
                output = x_train[j * batch_size: (j + 1) * batch_size]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                err += self.loss(y_train[j * batch_size: (j + 1) * batch_size], output)
                n_correct += num_correct(pred=output, real=y_train[j * batch_size: (j + 1) * batch_size])

                error = y_train[j * batch_size: (j + 1) * batch_size]
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error)
            err /= batch_size
            error_list.append(err)
            acc_list.append(n_correct / len(x_train))
            self.decay_learning_rate(.99)

            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
        plt.plot(error_list)
        plt.title('swish roo eroo')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

        plt.plot(acc_list)
        plt.title('Swiss Roll Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        print(f"first acc: {acc_list[0]}")
        print(f"last acc: {acc_list[-1]}")

    def decay_learning_rate(self, factor):
        for layer in self.layers:
            layer.learning_rate *= factor
