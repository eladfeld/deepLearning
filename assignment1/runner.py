from NN import NN, graph_err_and_acc_by_epoch
from DataLoader import DataLoader

swiss_roll = {
    'name': "Swiss Roll",
    'path': './data/SwissRollData.mat',
    'hidden_dims': [10, 20, 50, 20, 10],
    'lr': 0.0001,
    'epochs': 150,
    'batch_size': 10000
}

peaks = {
    'name': "Peaks",
    'path': './data/PeaksData.mat',
    'hidden_dims': [10, 20, 50, 20, 10],
    'lr': 0.0001,
    'epochs': 150,
    'batch_size': 10000
}

gmm = {
    'name': "GMM",
    'path': './data/GMMData.mat',
    'hidden_dims': [10, 20, 50, 20, 10],
    'lr': 0.0001,
    'epochs': 150,
    'batch_size': 10000
}

resnet_swissroll = {
    'name': "Res Swiss Roll",
    'path': './data/SwissRollData.mat',
    'hidden_dims': [30, 30, 30],
    'lr': 0.00001,
    'epochs': 500,
    'batch_size': 10000,
    'learning_rate_decay': 0.99,
    'resnet': True
}

data = resnet_swissroll

DATA_NAME = data['name']

#Data
dl = DataLoader(data['path'])
X = dl.training_inputs.T
Y = dl.training_labels.T
Xv = dl.validation_inputs.T
Yv = dl.validation_labels.T

#Params
in_dim, out_dim = X.shape[1], Y.shape[1]
layer_dims = [in_dim] + data['hidden_dims'] + [out_dim]
LR = data['lr']
EPOCHS = data['epochs']
BATCH_SIZE = data['batch_size']
resnet = 'resnet' in data and data['resnet']
learning_rate_decay = .99 if 'learning_rate_decay' not in data else data['learning_rate_decay']


print(f"in dim: {in_dim}\nout dim: {out_dim}\ntraining samples: {X.shape[0]}\nvalidation samples: {Xv.shape[0]}")

#Run
net = NN(layer_dims=layer_dims, learning_rate=LR, Xv=Xv, Yv=Yv, learning_rate_decay=learning_rate_decay, resnet=resnet)
err_train, acc_train, err_val, acc_val = net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
graph_err_and_acc_by_epoch(name=f"{DATA_NAME} Training", err=err_train, acc=acc_train)
graph_err_and_acc_by_epoch(name=f"{DATA_NAME} Validation", err=err_val, acc=acc_val)