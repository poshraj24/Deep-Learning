import numpy as np 
from Layers.Base import *
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], -1)) #(1st dimension-batch size, 2nd dimension- flattened product)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)