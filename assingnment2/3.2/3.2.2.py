import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 1000
HIDDEN_SIZE = 100
NUM_EPOCHS = 35
RECON_TO_PRED_LOSS_RATIO = 30

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_data, test_data = torch.utils.data.random_split(mnist_data, [50000, 10000])


class AutoEncoder(nn.Module):
    def __init__(self, input_size, z_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 20
        self.encoder = nn.LSTM(input_size=28, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=28 * 28, num_layers=1, batch_first=True)
        self.U = nn.Linear(in_features=28 * 28, out_features=10)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 28, 28)
        out, hidden = self.encoder(x)
        out = out[:, -1:, :]
        out, (h, c) = self.decoder(out)
        prediction = self.U(h)
        prediction = self.softmax(prediction)
        return out, prediction


def labalizer(labels):
    one_hots = np.zeros((len(labels), 10))
    for index in range(len(labels)):
        one_hots[index][labels[index]] = 1
    return torch.Tensor(one_hots)


def accuracy(labels, preds):
    return np.sum(np.argmax(labels, axis=1) == np.argmax(preds, axis=1)) / labels.shape[0]


def train():
    recon_losses = []
    pred_losses = []
    accs = []
    for epoch in range(NUM_EPOCHS):
        for (imgs, labels) in train_dl:
            imgs = imgs.reshape(-1, 28 * 28)
            recons, preds = model(imgs)
            recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2])
            preds = preds.reshape(preds.shape[1], preds.shape[2])
            recon_loss = critertion(recons, imgs)
            one_hots = labalizer(labels)
            class_loss = critertion1(preds, one_hots)
            total_loss = recon_loss * RECON_TO_PRED_LOSS_RATIO + class_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # validation
        for imgs, labels in test_dl:
            recons, pred = model(imgs)
            recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2])
            imgs = imgs.reshape(imgs.shape[0], imgs.shape[2] * imgs.shape[3])
            pred = pred.reshape(pred.shape[1], pred.shape[2])
            recon_loss = critertion(recons, imgs)
            one_hot = labalizer(labels)
            class_loss = critertion1(pred, one_hot)
            acc = accuracy(one_hot.detach().numpy(), pred.detach().numpy())
            accs.append(acc)
            pred_losses.append(class_loss.detach().numpy())
            recon_losses.append(recon_loss.detach().numpy())
            print(f'Epoch:{epoch + 1}, acc = {acc:.3f}, recon_loss:{recon_loss:.4f}, class_loss:{class_loss:.4f}')
    validataion_graph_render(accs, pred_losses, recon_losses)


def validataion_graph_render(accs, pred_losses, recon_losses):
    plt.plot(accs, color='green')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('accuracy per epoch')
    plt.show()

    plt.plot(pred_losses, color='red')
    plt.title('prediction loss (cross entropy)')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

    plt.plot(recon_losses, color='orange')
    plt.title('reconstruction loss (MSE)')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

def test():
    for imgs, labels in test_dl:
        recons, pred = model(imgs)
        recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2]).detach().numpy()
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[2] * imgs.shape[3])
        plt.figure(figsize=(9, 2))
        plt.gray()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])
        for i, item in enumerate(recons):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])
        plt.show()



train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=True)

model = AutoEncoder(28 * 28, 12)
critertion = nn.MSELoss()
critertion1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)
train()
test()








# for k in range(0, NUM_EPOCHS, 4):
#     plt.figure(figsize=(9, 2))
#     plt.gray()
#     imgs = outputs[k][1].detach().numpy()
#     recons = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= 9: break
#         plt.subplot(2, 9, i + 1)
#         item = item.reshape(-1, 28, 28)
#         plt.imshow(item[0])
#
#     for i, item in enumerate(recons):
#         if i >= 9: break
#         plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
#         item = item.reshape(-1, 28, 28)
#         plt.imshow(item[0])
#
#     plt.show()
