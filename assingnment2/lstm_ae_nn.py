import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 1000
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=BATCH_SIZE, shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))

class AutoEncoder(nn.Module):
    def __init__(self, input_size, z_size):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 20
        self.encoder = nn.LSTM(input_size=28, hidden_size=4, num_layers=4, batch_first=True)
        self.decoder = nn.LSTM(input_size=4, hidden_size=28, num_layers=4, batch_first=True)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Passing in the input and hidden state into the model and  obtaining outputs
        x = x.reshape(1000, 28, 28)
        out, hidden = self.encoder(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, hidden = self.decoder(out)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = self.fc(out[:, -1, :])
        return out

model =AutoEncoder(28*28, 12)
critertion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


NUM_EPOCHS = 21
outputs = []

for epoch in range(NUM_EPOCHS):
    for(img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        recon = model(img)
        recon = recon.reshape(recon.shape[0], recon.shape[1]*recon.shape[2])
        loss = critertion(recon, img)
        # loss = loss.reshape(loss[1], loss[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))

for k in range(0, NUM_EPOCHS, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i + 1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    plt.show()

