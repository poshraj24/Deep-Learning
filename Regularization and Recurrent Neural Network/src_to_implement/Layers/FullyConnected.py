import sys
import os
from pathlib import Path
project_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_path))
sys.path.append(os.getcwd())
import numpy as np
from Layers.Base import *
from Layers.Initializers import *
from Optimization.Optimizers import *
class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.weights = np.random.uniform(size=(input_size+1, output_size))
        #self.biases = np.random.rand(1, output_size)
        self.trainable=True
        self._optimizer=None
        self.gradient_tensor=None
        self.input_tensor=None
        self.gradient_weights=None

    #added initializer for CNN
    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        initialize_bias = lambda initializer, size: initializer.initialize((1, size), 1, size)
        bias = initialize_bias(bias_initializer, self.output_size)
        self.weights = np.r_[weights, bias]

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, sett):
        self._optimizer = sett
    
    
    #Forward Pass
    def forward(self, input_tensor):
               
        self.input_tensor = np.insert(input_tensor, input_tensor.shape[1], 1, axis=1)
        #self.input_tensor=input_tensor

        # Calculate output values from inputs, weights and biases
        self.output = np.dot(self.input_tensor, self.weights) #+ self.biases
        return self.output
    
    
    # Backward pass
    def backward(self, error_tensor):
        # Gradients on parameters
        self.error_tensor = error_tensor @ self.weights[:-1, :].T
        
        gradient_tensor = self.input_tensor.T @ error_tensor
        #gradient on values
        self.dinputs = error_tensor @ self.weights[:-1, :].T
        self.gradient_weights = gradient_tensor

        
        
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
            #self.biases = self.optimizer.calculate_update(self.biases, np.mean(error_tensor, axis=0, keepdims=True))
        return self.error_tensor
    
    