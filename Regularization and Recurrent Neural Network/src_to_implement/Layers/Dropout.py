import numpy as np

from Layers.Base import *


class Dropout(BaseLayer):

    def __init__(self, probability):
        super().__init__()

        self.prob = probability
        self.mask = None

    def forward(self, input_tensor):
        # if testing no need to
        self.mask = np.random.rand(*input_tensor.shape) < self.prob if not self.testing_phase else np.ones_like(input_tensor)
        return input_tensor if self.testing_phase else (input_tensor * self.mask) / self.prob


    def backward(self, error_tensor):

        return (self.mask*error_tensor  ) / self.prob


