import numpy as np
from Layers.SoftMax import *
class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.input = prediction_tensor
    
    # Add a small epsilon to avoid numerical instability (log(0))
        epsilon = np.finfo(float).eps
    
    # Compute the negative log likelihood
        loss = -np.sum(label_tensor * np.log(prediction_tensor + epsilon))
    
        return loss

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
    
    # Compute the gradient of the loss with respect to the input
        error_tensor = -label_tensor / (self.input + epsilon)
    
        return error_tensor
    
   
