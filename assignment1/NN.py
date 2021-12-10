import matplotlib.pyplot as plt
import numpy as np

from Layer import Layer
from OutputLayer import OutputLayer
from SGD import unison_shuffled_copies
from ResLayer import ResLayer

#region Functions


def cross_entropy(real, pred):
    (m, n) = pred.shape
    log = np.log(pred)
    real_log = real * log
    numerate = np.sum(real_log,  axis=1)
    numerate = np.sum(numerate)
    return numerate / (-m)


def cross_prime(reals, preds):
    return preds - reals


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


def num_correct(pred, real):
    if len(pred.shape) != 2:
        print(f"pred shape: {pred.shape}")
    (n, m) = pred.shape
    p = np.argmax(pred, axis=1)
    r = np.argmax(real, axis=1)
    n_correct = np.sum(p == r)
    return n_correct


def graph_epochs(name, y_label, y):
    plt.plot(y)
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.show()


def graph_err_and_acc_by_epoch(name, err, acc):
    graph_epochs(name, "Error",  err)
    graph_epochs(name, "Accuracy",  acc)


#endregion

class NN:
    def __init__(self, layer_dims, learning_rate, Xv, Yv, learning_rate_decay=0.99, resnet=False):
        self.learning_rate_decay = learning_rate_decay
        self.layers = []
        self.loss = cross_entropy
        self.d_loss = cross_prime
        self.init_layers(layer_dims, learning_rate, resnet)
        self.layers.append((OutputLayer(layer_dims[-2], layer_dims[-1], learning_rate)))
        self.X_validation = Xv
        self.Y_validation = Yv
    def init_layers(self, layer_dims, learning_rate, resnet):
        if not resnet:
            for i in range(len(layer_dims) - 2):
                self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], learning_rate))
        else:
            for i in range(len(layer_dims) - 2):
                if layer_dims[i] == layer_dims[i + 1]:
                    self.layers.append(ResLayer(layer_dims[i], layer_dims[i + 1], learning_rate))
                else:
                    self.layers.append(Layer(layer_dims[i], layer_dims[i + 1], learning_rate))


    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

    def fit(self, x_train, y_train, epochs, batch_size):
        samples = len(x_train)
        error_list, acc_list = [], []
        v_err, v_acc = [], []
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
            v_err, v_acc = self.validate(v_err, v_acc)
            self.decay_learning_rate(self.learning_rate_decay)
            # print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
        return error_list, acc_list, v_err, v_acc

    def validate(self, v_err, v_acc):
        X, Y = self.X_validation, self.Y_validation
        preds = self.predict(X)
        err = self.loss(Y, preds)
        acc = num_correct(np.array(preds), Y) / len(X)
        v_err.append(err)
        v_acc.append(acc)
        return v_err, v_acc

    def decay_learning_rate(self, factor):
        for layer in self.layers:
            layer.learning_rate *= factor
