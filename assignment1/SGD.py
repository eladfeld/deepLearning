import numpy as np
from matplotlib import pyplot as plt
from DataLoader import DataLoader

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


def sgd_linear(learning_rate, batch_size):
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
        m -= learning_rate * mse_grad_wrm(preds, reals, inputs)
        b -= learning_rate * mse_grad_wrb(preds, reals)
    plt.plot(X, Y, 'ro')
    plt.plot([0, 2], [linear_function(0), linear_function(2)])
    plt.plot([0, 2], [f(0), f(2)], '--')
    plt.show()


def softmax(z):
    e_x = np.exp(z - np.max(z))
    sum = np.sum(e_x, axis=1, keepdims=True)
    output = e_x / sum
    return output

def accuracy(pred, real):
    (n, m) = pred.shape
    p = np.argmax(pred, axis=1)
    r = np.argmax(real, axis=1)
    n_correct = np.sum(p == r)
    return n_correct / n

def cross_entropy(pred, real):
    (n, m) = pred.shape
    numerate = (np.sum(real * np.log(pred), axis=1))
    return (numerate / (-m))[0]


def ce_grad_wrw(preds, reals, inputs):
    return np.dot((preds - reals).T, inputs)


def ce_grad_wrb(preds, reals):
    return np.dot((preds - reals).T, (np.ones((preds.shape[0], 1)) / preds.shape[0]))


def sgd_softmax(learning_rate, batch_size, epochs, X, Y, X_test, Y_test, sample_num):
    w = np.random.normal(size=(Y.shape[1], X.shape[1]))
    b = np.zeros((Y.shape[1], 1))
    error_train_list, error_test_list, acc_train_list, acc_test_list = [], [], [], []
    for epoch in range(epochs):
        for i in range(int(len(X)/ batch_size)):
            inputs = X[i * batch_size: (i + 1) * batch_size]
            reals = Y[i * batch_size: (i + 1) * batch_size]
            z = (np.dot(inputs, w).T + b).T
            preds = softmax(z)
            w -= learning_rate * ce_grad_wrw(preds, reals, inputs)
            b -= learning_rate * ce_grad_wrb(preds, reals)
        idx_train = np.random.randint(len(X), size=sample_num)
        idx_test = np.random.randint(len(X_test), size=sample_num)
        error_train, acc_train, error_test, acc_test = validate(w, b, X[idx_train], Y[idx_train], Y_test[idx_test], Y_test[idx_test])
        error_train_list += error_train
        error_test_list += error_train
        acc_test_list += acc_test
        acc_train_list += acc_train
    plot_results(error_train_list, error_test_list, acc_train_list, acc_test_list)


def plot_results(error_train_list, error_test_list, acc_train_list, acc_test_list):
    pass

def validate(w, b, X, Y, X_test, Y_test):
    z = (np.dot(X, w).T + b).T
    preds = softmax(z)
    error = cross_entropy(preds, Y)
    acc = accuracy(preds, Y)
    return error_train, acc_train, error_test, acc_test

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

dl = DataLoader('./data/SwissRollData.mat')
X = dl.training_inputs.T[0: 10000]
Y = dl.training_labels.T[0: 10000]

# X, Y = unison_shuffled_copies(X, Y)
# sgd_softmax(0.01, 100, 100, X, Y)
# sgd_linear(0.1, 200)
