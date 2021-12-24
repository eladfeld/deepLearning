import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from assingnment2.lstm_ae_toy import function_that_prints_the_graphs_but_with_extra_words_in_the_name

stocks = pd.read_csv('data/SP 500 Stock Prices 2014-2017.csv')
stocks = stocks.sort_values(['symbol', 'date']).drop(['open', 'close', 'low', 'volume', 'date'], axis=1)
amzn = stocks[stocks.symbol == 'AMZN'].drop(['symbol'], axis=1)
googl = stocks[stocks.symbol == 'GOOGL'].drop(['symbol'], axis=1)


amzn = torch.tensor(amzn['high'].values).float()
googl = torch.tensor(googl['high'].values).float()

def print_graphs(stocks, title):
    plt.plot(stocks)
    plt.xlabel('stock market days since 2014-01-02')
    plt.ylabel('daily high')
    plt.title(title)
    plt.show()

def split_data_into_seqs_partions(stocks, sequnce_length):
    print_graphs(stocks.detach().numpy(), "pre split data")
    tuples = torch.split(stocks, sequnce_length)#[0:-1]
    tnsr = torch.stack(tuples)
    print_graphs(tnsr.detach().numpy(), "split data")
    return tnsr



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_STATE_SIZE, num_layers=NUM_LAYERS,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=HIDDEN_STATE_SIZE, hidden_size=INPUT_SIZE, num_layers=NUM_LAYERS,
                               batch_first=True)

    def forward(self, x):
        integer = int(SEQUENCE_SIZE / INPUT_SIZE)
        x = x.reshape(x.shape[0], integer, INPUT_SIZE)
        out, hidden = self.encoder(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, hidden = self.decoder(out)
        return out

# HP
BATCH_SIZE = 50
HIDDEN_STATE_SIZE = 10
LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = 1e-4

# constant
NUM_LAYERS = 2
WEIGHTS_DECAY = 1e-5
NUM_EPOCHS = 200
SEQUENCE_SIZE = 19
INPUT_SIZE = SEQUENCE_SIZE

google_seqs = split_data_into_seqs_partions(googl, SEQUENCE_SIZE) /1200
amzn_seqs = split_data_into_seqs_partions(amzn, SEQUENCE_SIZE) /1200

NUM_SEQUENCES = len(google_seqs)

model = AutoEncoder()
critertion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

outputs = []

def train(data):
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        recons = []
        for i in range(0, data.shape[0], BATCH_SIZE):
            seq = data[i: i + BATCH_SIZE]
            recon = model(seq).reshape(BATCH_SIZE, SEQUENCE_SIZE)
            loss = critertion(recon, seq)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recons += list(recon.flatten().detach().numpy())
        outputs.append(epoch_loss)
        print(f'epoch: {epoch}\tepoch_loss: {epoch_loss}')
    print_graphs(data.detach().numpy(), "original")
    print_graphs(recons, "recon")



train(google_seqs)
outputs = [x.detach().numpy() for x in outputs]
plt.plot(outputs)
plt.show()