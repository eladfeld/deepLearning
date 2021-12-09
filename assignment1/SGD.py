import numpy as np
from matplotlib import pyplot as plt
from DataLoader import DataLoader
import data_maker as dm



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
    m = np.max(z, axis=0, keepdims=True)
    ZminusM  = z - m
    e_x = np.exp(ZminusM)
    sum_ = np.sum(e_x, axis=0, keepdims=True)
    output = e_x / sum_
    return output


def accuracy(pred, real):
    (n, m) = pred.shape
    p = np.argmax(pred, axis=1)
    r = np.argmax(real, axis=1)
    n_correct = np.sum(p == r)
    return n_correct / n


def cross_entropy(pred, real):
    (n, m) = pred.shape
    log = np.log(pred)
    real_log = real.T * log
    numerate = np.sum(real_log,  axis=1)
    numerate = np.sum(numerate)
    return numerate / (-m)


def ce_grad_wrw(preds, reals, inputs):
    return np.dot((preds - reals.T), inputs)


def ce_grad_wrb(preds, reals):
    PminusR = preds - reals.T
    mean = np.mean(PminusR, axis=1, keepdims=True)
    return mean


def sgd_softmax(learning_rate, batch_size, epochs, X, Y, X_test, Y_test, sample_num):
    w = np.random.normal(size=(Y.shape[1], X.shape[1]))
    b = np.zeros((Y.shape[1], 1))
    error_train_list, error_test_list, acc_train_list, acc_test_list = [], [], [], []
    for epoch in range(epochs):
        X, Y = unison_shuffled_copies(X, Y)
        for i in range(int(len(X) / batch_size)):
            inputs = X[i * batch_size: (i + 1) * batch_size]
            reals = Y[i * batch_size: (i + 1) * batch_size]
            z0 = np.dot(w, inputs.T)
            z = (z0 + b)
            preds = softmax(z)
            dw = ce_grad_wrw(preds, reals, inputs)
            db = ce_grad_wrb(preds, reals)
            w -= learning_rate * dw
            b -= (learning_rate * db)
            learning_rate *= .9
            # print('dw ', dw)
            # print('db', db)
            # print('input ', inputs)
            # print('reals ', reals)
            # print('preds', preds)
        error_train, acc_train, error_test, acc_test = validate(w, b, X, Y, X_test, Y_test, sample_num)
        error_train_list.append(error_train)
        error_test_list.append(error_test)
        acc_test_list.append(acc_test)
        acc_train_list.append(acc_train)
    plot_results(error_train_list, error_test_list, acc_train_list, acc_test_list)


def plot_results(error_train_list, error_test_list, acc_train_list, acc_test_list):
    print(error_train_list)
    plt.plot(error_train_list)
    plt.title('Train Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

    plt.plot(error_test_list)
    plt.title('Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    # plt.plot(acc_test_list)
    # plt.show()
    # plt.plot(acc_train_list)
    # plt.show()


def validate(w, b, X_train, Y_train, X_test, Y_test, sample_num):
    idx_train = np.random.randint(len(X_train), size=sample_num)
    idx_test = np.random.randint(len(X_test), size=sample_num)
    X_train, Y_train = X_train[idx_train], Y_train[idx_train]
    X_test, Y_test = X_test[idx_test], Y_test[idx_test]

    z_train = np.dot(w, X_train.T) + b
    z_test = np.dot(w, X_test.T) + b

    preds_test = softmax(z_test)
    preds_train = softmax(z_train)
    error_train = cross_entropy(preds_train, Y_train)
    error_test = cross_entropy(preds_test, Y_test)
    acc_train = accuracy(preds_train, Y_train)
    acc_test = accuracy(preds_test, Y_test)
    return error_train, acc_train, error_test, acc_test

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def check():
    dl = DataLoader('./data/SwissRollData.mat')
    X_train = dl.training_inputs.T
    Y_train = dl.training_labels.T
    X_test = dl.validation_inputs.T
    Y_test = dl.validation_labels.T
    # X_train, Y_train = dm.make_swirl_data(20000)
    # X_test, Y_test = dm.make_swirl_data(50)
    x = [X_train[i][0] for i in range(len(X_train))]
    y = [X_train[i][1] for i in range(len(X_train))]

    plt.plot(x, y, 'o')
    plt.show()
    sgd_softmax(0.01, 10000, 25, X_train, Y_train, X_test, Y_test, 100)
    # sgd_softmax(0.1, 1, 50, X_train, Y_train, X_test, Y_test, 50)

    # sgd_linear(0.1, 200)

# check()
