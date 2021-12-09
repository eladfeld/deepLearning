from scipy.io import loadmat
import numpy as np

# DATA FILE PATHS
# 'assignment1//data/SwissRollData.mat'
# 'assignment1//data/PeaksData.mat'
# 'assignment1//data/GMMData.mat'

TRAINING_INPUT_KEY = 'Yt'
TRAINING_LABEL_KEY = 'Ct'
VALIDATION_INPUT_KEY = 'Yv'
VALIDATION_LABEL_KEY = 'Cv'


# TODO: consider using numpy arrays/vectors
# TODO: implement for validation data

class DataLoader:
    def __init__(self, path):
        data = loadmat(path)
        self.data = data  # TODO: remove
        self.input_dim = len(self.data[TRAINING_INPUT_KEY])
        self.num_labels = len(self.data[TRAINING_LABEL_KEY])
        self.training_inputs = data[TRAINING_INPUT_KEY]
        self.training_labels = data[TRAINING_LABEL_KEY]
        self.validation_inputs = data[VALIDATION_INPUT_KEY]
        self.validation_labels = data[VALIDATION_LABEL_KEY]

    def get_training_data_size(self):
        return len(self.training_inputs[0])

    def get_training_input_point(self, index):
        output = [float('NaN')] * self.input_dim
        for dim in range(0, self.input_dim):
            output[dim] = self.training_inputs[dim][index]
        return output

    # returns label in range [0, numLabels)
    def get_training_point_label(self, index):
        for label in range(0, self.num_labels):
            if self.training_labels[label][index] == 1:
                return label
