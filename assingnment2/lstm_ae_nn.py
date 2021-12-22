import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def labalizer(labels):
    one_hots = np.zeros((len(labels), 10))
    for index in range(len(labels)):
        one_hots[index][labels[index]] = 1
    return torch.Tensor(one_hots)


def accuracy(labels, preds):
    return np.sum(np.argmax(labels, axis=1) == np.argmax(preds, axis=1)) / labels.shape[0]

BATCH_SIZE = 1000
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

print(len(mnist_data))
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=BATCH_SIZE, shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))


class AutoEncoder(nn.Module):
    def __init__(self, input_size, z_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 20
        self.encoder = nn.LSTM(input_size=28, hidden_size=4, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=4, hidden_size=28, num_layers=1, batch_first=True)
        self.U = nn.Linear(in_features=28, out_features=10)
        self.softmax = nn.Softmax(dim=2)



    def forward(self, x):
        x = x.reshape(BATCH_SIZE, 28, 28)
        out, hidden = self.encoder(x)
        out, (h, c) = self.decoder(out)
        prediction = self.U(h)
        prediction = self.softmax(prediction)
        return out, prediction



model = AutoEncoder(28 * 28, 12)
critertion = nn.MSELoss()
critertion1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

NUM_EPOCHS = 50
outputs = []
graphim = []

for epoch in range(NUM_EPOCHS):
    for (imgs, labels) in data_loader:
        imgs = imgs.reshape(-1, 28 * 28)
        recons, preds = model(imgs)
        recons = recons.reshape(recons.shape[0], recons.shape[1] * recons.shape[2])
        preds = preds.reshape(preds.shape[1], preds.shape[2])
        recon_loss = critertion(recons, imgs)
        one_hots = labalizer(labels)
        class_loss = critertion1(preds, one_hots)
        total_loss = recon_loss * 50 + class_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    acc = accuracy(one_hots.detach().numpy(), preds.detach().numpy())

    print(f'Epoch:{epoch + 1}, Loss:{total_loss.item():.4f}, acc = {acc:.3f}, recon_loss:{recon_loss:.4f}, class_loss:{class_loss:.4f}')
    outputs.append((epoch, imgs, recons))


for k in range(0, NUM_EPOCHS, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recons = outputs[k][2].detach().numpy()
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
