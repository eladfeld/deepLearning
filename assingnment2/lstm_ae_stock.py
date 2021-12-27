import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


stocks = pd.read_csv('data/SP 500 Stock Prices 2014-2017.csv')
stocks = stocks.sort_values(['symbol', 'date']).drop(['open', 'close', 'low', 'volume', 'date'], axis=1)


amzn = stocks[stocks.symbol == 'AMZN'].drop(['symbol'], axis=1)
googl = stocks[stocks.symbol == 'GOOGL'].drop(['symbol'], axis=1)
amzn = torch.tensor(amzn['high'].values).float()
googl = torch.tensor(googl['high'].values).float()

names = stocks['symbol'].unique().tolist()
all_stocks = []
for name in names:
    to_add = stocks[stocks.symbol == name].drop(['symbol'], axis=1)
    to_add = torch.tensor(to_add['high'].values).float()
    all_stocks.append(to_add)

#data preprocessing - remove short tensor and tensors with Nan values
all_stocks = [x for x in all_stocks if x.shape[0] > 1000]
nan_indices = []
for stock_idx in reversed(range(len(all_stocks))):
    if torch.isnan(all_stocks[stock_idx]).any():
        del names[stock_idx]
        del all_stocks[stock_idx]



limit = int(len(all_stocks) * 0.8)
train_data = all_stocks[:limit]
test_data = all_stocks[limit:]


def print_graphs(stocks, title):
    plt.plot(stocks)
    plt.xlabel('stock market days since 2014-01-02')
    plt.ylabel('daily high')
    plt.title(title)
    plt.show()


def split_data_into_seqs_partions(stocks, sequnce_length):
    tuples = torch.split(stocks, sequnce_length)[0:-1]
    tnsr = torch.stack(tuples)
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
BATCH_SIZE = 10
HIDDEN_STATE_SIZE = 10
LEARNING_RATE = 1e-3

# constant
NUM_LAYERS = 1
NUM_EPOCHS = 30
SEQUENCE_SIZE = 100
INPUT_SIZE = 1


def split_data(data):
    spliited_data = split_data_into_seqs_partions(data, SEQUENCE_SIZE)
    return spliited_data


def normalize_data(data):
    m = torch.min(data).detach().numpy()
    M = torch.max(data).detach().numpy()
    normlized_data = (data - m) / (M - m)
    denormlize = lambda data: data * (M - m) + m
    return normlized_data, denormlize


model = AutoEncoder()
critertion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(stocks):
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for stock in stocks:
            data = split_data(stock)
            data, _ = normalize_data(data)
            for i in range(0, data.shape[0], BATCH_SIZE):
                seq = data[i: i + BATCH_SIZE]
                recon = model(seq).reshape(BATCH_SIZE, SEQUENCE_SIZE)
                loss = critertion(recon, seq)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f'epoch: {epoch}\tepoch_loss: {epoch_loss}')


def test(stocks):
    losses = []
    index = limit
    for stock in stocks:
        data = split_data(stock)
        data, denormalize = normalize_data(data)
        recon = model(data).reshape(data.shape)
        loss = critertion(recon, data)
        losses.append(loss)
        graph_render(denormalize(data.detach().numpy().flatten()), denormalize(recon.detach().numpy().flatten()), names[index])
        index += 1

def graph_render(orig, recon, data_name):
    plt.title(f'{data_name} stocks lr:{LEARNING_RATE}')
    plt.xlabel('time')
    plt.ylabel('daily high')
    plt.plot(orig, label='input')
    plt.plot(recon, label='reconstruction')
    plt.legend()
    plt.show()


train(train_data)

test(test_data)


