import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from lstm_ae_toy import data_maker, SEQUENCE_SIZE, NUM_SEQUENCES, \
    function_that_prints_the_graphs_but_with_extra_words_in_the_name

transform = transforms.ToTensor()

# HP
BATCH_SIZE = 1000
HIDDEN_STATE_SIZE = 4
LEARNING_RATE = 1e-3
GRADIENT_CLIPPING = 1e-4

# constant
INPUT_SIZE = 1
NUM_LAYERS = 1
WEIGHTS_DECAY = 1e-5
NUM_EPOCHS = 200

dm = data_maker()
train_and_validation_data, test_data = dm.make()


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


model = AutoEncoder()
critertion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHTS_DECAY)

outputs = []

for epoch in range(NUM_EPOCHS):
    limit = int((epoch / NUM_EPOCHS) * (NUM_SEQUENCES * .6))
    valid_size = int(NUM_SEQUENCES * .2)
    train_data = torch.cat((train_and_validation_data[:limit], train_and_validation_data[limit + valid_size:]), 0)
    validation_data = train_and_validation_data[limit:limit + valid_size]
    #train
    for i in range(0, train_data.shape[0], BATCH_SIZE):
        seq = train_data[i: i + BATCH_SIZE]
        recon = model(seq).reshape(BATCH_SIZE, SEQUENCE_SIZE)
        # recon = recon.reshape(recon.shape[0], recon.shape[1]*recon.shape[2])
        loss = critertion(recon, seq)
        # loss = loss.reshape(loss[1], loss[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #check validation
    recon = model(validation_data).reshape(validation_data.shape[0], SEQUENCE_SIZE)
    v_loss = critertion(recon, validation_data)
    # loss = loss.reshape(loss[1], loss[2])
    optimizer.zero_grad()

    print(f'Epoch:{epoch + 1}, train - Loss:{loss.item():.4f} .... validation - loss:{v_loss.item():.4f}')
    recon = model(test_data).reshape(test_data.shape[0], SEQUENCE_SIZE)
    t_loss = critertion(recon, test_data)
    optimizer.zero_grad()
    print(f"this is the final loss for the training period, mmmm sir: {t_loss}")

recon = model(test_data).reshape(test_data.shape[0], SEQUENCE_SIZE)

for i in range(3):
    function_that_prints_the_graphs_but_with_extra_words_in_the_name(test_data.detach().numpy(), i, 'synthetic input')
    function_that_prints_the_graphs_but_with_extra_words_in_the_name(recon.detach().numpy(), i, 'synthetic reconstruction')

