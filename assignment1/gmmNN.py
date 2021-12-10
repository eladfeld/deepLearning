from NN import NN, graph_err_and_acc_by_epoch
from DataLoader import DataLoader

DATA_NAME = "GMM"

#Data
dl = DataLoader('./data/GMMData.mat')
X = dl.training_inputs.T
Y = dl.training_labels.T
Xv = dl.validation_inputs.T
Yv = dl.validation_labels.T


#Params
in_dim, out_dim = X.shape[1], Y.shape[1]
layer_dims = [in_dim, 10, 20, 50, 20, 10, out_dim]
LR = 0.0001
EPOCHS = 150
BATCH_SIZE = 10000

print(f"in dim: {in_dim}\nout dim: {out_dim}\ntraining samples: {X.shape[0]}\nvalidation samples: {Xv.shape[0]}")

net = NN(layer_dims=layer_dims, learning_rate=LR, Xv=Xv, Yv=Yv)

err_train, acc_train, err_val, acc_val = net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

graph_err_and_acc_by_epoch(name=f"{DATA_NAME} Training", err=err_train, acc=acc_train)

graph_err_and_acc_by_epoch(name=f"{DATA_NAME} Validation", err=err_val, acc=acc_val)