import numpy as np
from matplotlib import pyplot as plt


# x.shape = (n, 1)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def cross_entropy(pred, real):
    (n, m) = pred.shape
    return (np.sum(real * np.log(pred), axis=1) / (-m))[0]


def softmax_grad(x):
    sm = softmax(x)
    s = sm.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def grad_test_softmax():
    F = cross_entropy
    g_F = softmax_grad
    n = 20
    x = np.random.randn(n)
    d = np.random.randn(n)
    epsilon = 0.1
    F0 = F(x)
    g0 = g_F(x)
    y0 = np.zeros(8)
    y1 = np.zeros(8)
    print("k\terror  oreder 1\t\t error order 2")
    for k in range(8):
        epsk = epsilon * (0.5 ** k)
        Fk = F(x + epsk * d)
        F1 = F0 + epsk * np.dot(g0, d)
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)
        print(f'{k} \t {y0[k]} \t {y1[k]}')
    return y0, y1


grad_test_softmax()
