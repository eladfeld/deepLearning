import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ROW_LENGTH = 28
NUM_PIXELS = ROW_LENGTH ** 2
BATCH_SIZE = 10000
HIDDEN_SIZE = 100
NUM_EPOCHS = 1
INPUT_SIZE = 1

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_data, test_data = torch.utils.data.random_split(mnist_data, [50000, 10000])


class AutoEncoder(nn.Module):
    def __init__(self, input_size, z_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 20
        self.encoder = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=NUM_PIXELS, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.reshape(x.shape[0], NUM_PIXELS // INPUT_SIZE, INPUT_SIZE)
        out, hidden = self.encoder(x)
        out = out[:, -1:, :]
        out, (h, c) = self.decoder(out)
        return out


def labalizer(labels):
    one_hots = np.zeros((len(labels), 10))
    for index in range(len(labels)):
        one_hots[index][labels[index]] = 1
    return torch.Tensor(one_hots)


def train():
    losses = []
    for epoch in range(NUM_EPOCHS):
        for (imgs, labels) in train_dl:
            imgs = imgs.reshape(-1, 28 * 28)
            recons = model(imgs)
            recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2])
            loss = critertion(recons, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        for imgs, labels in test_dl:
            recons= model(imgs)
            recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2])
            imgs = imgs.reshape(imgs.shape[0], imgs.shape[2] * imgs.shape[3])
            loss = critertion(recons, imgs)
            losses.append(loss.detach().numpy())
            print(f'Epoch:{epoch + 1}, loss:{loss:.4f}')

def test():
    for imgs, labels in test_dl:
        recons = model(imgs)
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