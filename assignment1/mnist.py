import numpy as np
from Layer import Layer
from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils

from NN import NN, graph_err_and_acc_by_epoch

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2])

net = NN([28*28, 100, 50, 50, 10], 0.05, x_test, y_test)
err_train, acc_train, err_val, acc_val = net.fit(x_train, y_train, epochs=30, batch_size=1000)
graph_err_and_acc_by_epoch(name=f"mnist Training", err=err_train, acc=acc_train)
graph_err_and_acc_by_epoch(name=f"mnist Validation", err=err_val, acc=acc_val)


# image = x_test[i]
# image = np.array(image, dtype='float')
# pixels = image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# print(real[i], "  -  ", predicts[i])
# plt.show()
