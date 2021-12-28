import numpy as np
import matplotlib.pyplot as plt
import torch

NUM_SEQUENCES = 10000
SEQUENCE_SIZE = 50
LOW = 20
HIGH = 30


class data_maker():
    def make(self):
        data = np.random.rand(NUM_SEQUENCES, SEQUENCE_SIZE)
        m = []
        for sequence in range(NUM_SEQUENCES):
            i = np.random.randint(low=LOW, high=HIGH)
            m.append(np.array([(0.1 if j in range(i - 5, i + 6) else 1.0) for j in range(SEQUENCE_SIZE)]))
        m = np.array(m)
        data = data * m
        train_limit = int(NUM_SEQUENCES * 0.8)
        return torch.Tensor(data[:train_limit]), torch.Tensor(data[train_limit:])


def function_that_prints_the_graphs_but_with_extra_words_in_the_name(input, reconstruction, index, title):
    plt.title(f'{title} - {index + 1}')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.plot(input[index], label='input')
    plt.plot(reconstruction[index], label='reconstruction')
    plt.legend()
    plt.show()

# dm = data_maker()
# data = dm.make()
#
# for i in range(3):
#     function_that_prints_the_graphs_but_with_extra_words_in_the_name(i)