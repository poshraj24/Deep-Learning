# Softmax activation
import numpy as np
from Layers.Base import *
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    # Forward pass
    def forward(self, input_tensor):
        # Remember input values
        self.input_tensor= input_tensor
        # Get unnormalized probabilities
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


    def backward(self, error_tensor):
        errortensor_alias = error_tensor * self.output
        et_sum = errortensor_alias.sum(axis=1)
        et_sum = et_sum[:, np.newaxis]
        errortensor_alias = self.output * (error_tensor - et_sum)
        return errortensor_alias     
# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, input_tensor, error_tensor):
        # Calculate sample losses
        sample_losses = self.forward(input_tensor, error_tensor)
    # Calculate mean loss
        data_loss = np.mean(sample_losses)
    # Return loss
        return data_loss
