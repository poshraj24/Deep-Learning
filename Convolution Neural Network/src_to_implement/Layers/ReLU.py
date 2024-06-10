import numpy as np
from Layers.Base import *
class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    # Forward pass
    def forward(self, input_tensor):
        self.input_tensor=input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output
    
    # Backward pass
    def backward(self, error_tensor):
        self.dinputs = error_tensor.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.input_tensor <= 0] = 0
        return self.dinputs
        
