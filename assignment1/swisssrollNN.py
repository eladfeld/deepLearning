import numpy as np
from NN import NN
from Layer import Layer
from matplotlib import pyplot as plt

from assignment1.DataLoader import DataLoader


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


dl = DataLoader('./data/SwissRollData.mat')
X = dl.training_inputs.T
Y = dl.training_labels.T


net = NN(mse, mse_prime, [2, 10, 10, 2], 0.01)

net.fit(X, Y, epochs=100)



