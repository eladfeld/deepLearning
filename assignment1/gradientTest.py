import numpy as np
from matplotlib import pyplot as plt
from SGD import ce_grad_wrw, ce_grad_wrb, cross_entropy

def one_hot_vec(size, idx=0):
    output = np.zeros((size, 1))
    output[idx] = 1
    return output

def grad_test_softmax():
    n = 20
    x = np.abs(np.random.randn(n, 1))
    d = np.random.randn(n, 1) / n
    epsilon = 0.1
    r = one_hot_vec(n)

    F = lambda p: cross_entropy(p, r)
    g_F = lambda p: ce_grad_wrw(p, r, x)

    F0 = F(x)
    g0 = g_F(x.T)
    y0 = np.zeros(8)
    y1 = np.zeros(8)
    print("k\terror  oreder 1\t\t error order 2")
    for k in range(8):
        epsk = epsilon * (0.5 ** k)
        Fk = F(x + epsk * d)
        F1 = F0 + epsk * np.dot(g0.T, d)
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)
        print(f'{k} \t {y0[k]} \t {y1[k]}')
        plt.plot()
    return y0, y1


y0, y1 = grad_test_softmax()
plt.plot(y0)
plt.plot(y1)
plt.yscale('log')
plt.show()