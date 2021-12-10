import numpy as np
from NN import NN
from Layer import Layer
from matplotlib import pyplot as plt
from assignment1.DataLoader import DataLoader
import data_maker as data_maken


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


dl = DataLoader('./data/PeaksData.mat')
X = dl.training_inputs.T
Y = dl.training_labels.T
# X, Y = data_maken.argmax_data(10, 20000)

# Xv = dl.validation_inputs.T
# Yv = dl.validation_labels.T
in_dim, out_dim = X.shape[1], Y.shape[1]

net = NN(cross_entropy, cross_prime, [in_dim, 10, 20, 20, 10, out_dim], 0.0001)

net.fit(X, Y, epochs=200, batch_size=10000)
print(f"in dim: {in_dim}, out dim: {out_dim}")
# net.predict(Xv, Yv)



