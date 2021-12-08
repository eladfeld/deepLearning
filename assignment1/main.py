import numpy as np
from NN import NN
from Layer import Layer
from ActiveLayer import ActiveLayer


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = NN(mse, mse_prime)
net.add(Layer(2, 3, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))
net.add(Layer(3, 1, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))

net.fit(x_train, y_train, epochs=50)

print(net.predict(x_train))




