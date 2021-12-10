import numpy as np
from NN import NN, tanh, tanh_prime
from Layer import Layer
from ActiveLayer import ActiveLayer
from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;





(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


net = NN(mse, mse_prime)
net.add(Layer(28*28, 100, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))
net.add(Layer(100, 50, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))
net.add(Layer(50, 50, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))
net.add(Layer(50, 10, 0.1))
net.add(ActiveLayer(tanh, tanh_prime))
net.fit(x_train[0:5], y_train[0:5], epochs=2)

test_samples = 10
out = net.predict(x_test[0:test_samples])
print("\n")
print("predicted values : ")
predicts = []
for ou in out:
    predicts.append(np.argmax(ou))
real = []
for y in y_test[0:test_samples]:
    real.append(np.argmax(y))

correct = 0
mistakes = []
for i in range(len(out)):
    if predicts[i] == real[i]:
        correct += 1
    else:
        mistakes.append(i)

print(correct)
print(f'{correct * 100 / len(out)}% accuracy')
print(mistakes)
print(real)
for i in mistakes:
    image = x_test[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    print(real[i], "  -  ", predicts[i])
    plt.show()
