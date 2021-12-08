import numpy as np
from matplotlib import pyplot as plt


def linear_function(x):
    return 6 * x + 3


def noisy_linear_function(x):
    return linear_function(x) + np.random.normal(scale=0.8)


def mse(pred, real):
    x1 = real - pred
    x2 = x1 * x1
    return np.sum(x2) / pred.shape[0]


def mse_grad_wrm(preds, reals, inputs):
    return 2 * np.sum((preds - reals) * inputs) / len(preds)


def mse_grad_wrb(preds, reals):
    return 2 * np.sum(preds - reals) / len(preds)


def sgd(learning_rate, batch_size):
    X = np.array([x for x in np.arange(0, 2, 0.0001)])
    np.random.shuffle(X)
    Y = np.array([noisy_linear_function(x) for x in X])
    m = 1
    b = 0
    for i in range(int((len(X) / batch_size)) - 1):
        f = lambda x: x * m + b
        inputs = X[i * batch_size: (i + 1) * batch_size]
        reals = Y[i * batch_size: (i + 1) * batch_size]
        preds = f(inputs)
        error = mse(preds, reals)
        print(error)
        print("mse_grad_wrm", mse_grad_wrm(preds, reals, inputs))
        m -= learning_rate * mse_grad_wrm(preds, reals, inputs)
        b -= learning_rate * mse_grad_wrb(preds, reals)
        print(f'm: {m}, b:{b}')
    plt.plot(X, Y, 'ro')
    plt.plot([0, 2], [linear_function(0), linear_function(2)])
    plt.plot([0, 2], [f(0), f(2)], '--')
    plt.show()


sgd(0.1, 200)
