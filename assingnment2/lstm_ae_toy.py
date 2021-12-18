import numpy as np
import matplotlib.pyplot as plt
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
        return data

dm = data_maker()
data = dm.make()

for i in range(3):
    plt.title(f'synthetic data - example {i + 1}')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.plot(data[i])
    plt.show()