import numpy as np
from matplotlib import pyplot as plt
from SGD import ce_grad_wrw, ce_grad_wrb, cross_entropy, softmax

def one_hot_vec(size, idx=0):
    output = np.zeros((size, 1))
    output[idx] = 1
    return output

(out_dim, in_dim) = (3, 3)
_input = np.random.normal(size=(3, 1))
_real = one_hot_vec(out_dim)

def F(w):
    W = w.reshape((out_dim, in_dim))
    z = np.dot(W, _input)
    sm = softmax(z) #b can be anything, so zero
    return cross_entropy(sm, _real)

def g_F(w):
    W = w.reshape((out_dim, in_dim))
    z = W @ _input
    output_matrix = ce_grad_wrw(softmax(z), _real.T, _input.T)
    output_vec = output_matrix.reshape(9)
    return output_vec

def grad_test_softmax():
    n = in_dim * out_dim
    w = np.random.randn(n)
    d = np.random.randn(n)
    epsilon = 0.1
    F0 = F(w)
    g0 = g_F(w)
    y0 = np.zeros(8)
    y1 = np.zeros(8)
    print("k\terror  oreder 1\t\t error order 2")
    for k in range(8):
        epsk = epsilon * (0.5 ** k)
        Fk = F(w + epsk * d)
        F1 = F0 + epsk * np.dot(g0, d)
        y0[k] = np.abs(Fk - F0)
        y1[k] = np.abs(Fk - F1)
        print(f'{k} \t {y0[k]} \t {y1[k]}')
    return y0, y1


y0, y1 = grad_test_softmax()
plt.plot(y0)
plt.plot(y1)
plt.yscale('log')
plt.show()