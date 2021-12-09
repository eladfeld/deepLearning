import numpy as np

PI = np.pi


def argmax_data(dim=10, num_samples=25000):
    np.random.seed(42)
    inputs = np.random.rand(num_samples, dim)
    labels = np.random.rand(num_samples, dim)

    for i in range(len(labels)):
        c = np.argmax(inputs[i])
        for j in range(len(labels[0])):
            if j == c:
                labels[i][j] = 1
            else:
                labels[i][j] = 0

    return inputs, labels


def xor_data(dim=10, num_samples=25000):
    np.random.seed(42)
    inputs = np.random.randint(2, size=(num_samples, dim))
    labels = np.random.rand(num_samples, 2)

    for i in range(len(labels)):
        c = int(np.sum(inputs[i]) % 2)
        for j in range(2):
            if j == c:
                labels[i][j] = 1
            else:
                labels[i][j] = 0

    return inputs, labels


def make_swirl_data(n):
    inputs = []
    labels = []
    for t in np.arange(0, PI * 4, PI * 4 / n):
        x = t * np.cos(t) / 10
        y = t * np.sin(t) / 10
        inputs.append([x, y])
        labels.append([1.0, 0.0])

    for t in np.arange(0, PI * 4, PI * 4 / n):
        x = t * np.cos(t + PI) / 10
        y = t * np.sin(t + PI) / 10
        inputs.append([x, y])
        labels.append([0.0, 1.0])

    return np.array(inputs), np.array(labels)


def test_swirl_maker():
    inputs, labels = make_swirl_data(500)
    for i in range(len(inputs)):
        input = inputs[i]
        print(f"({input[0]}, {input[1]})")