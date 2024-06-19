import numpy as np

from Layers.Base import *

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape)
            return (input_tensor * self.mask) / self.probability

    def backward(self, error_tensor):
        return (error_tensor * self.mask) / self.probability