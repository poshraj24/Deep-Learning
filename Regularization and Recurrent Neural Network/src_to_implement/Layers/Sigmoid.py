import numpy as np
from Layers.Base import *

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out=None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor 
        return 1/(1 + np.exp(-input_tensor))

    def backward(self, error_tensor):
        return np.multiply(error_tensor, (1 - 1/(1 + np.exp(-self.input_tensor))) * 1/(1 + np.exp(-self.input_tensor)))