import numpy as np

from Layers.Base import *
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable=True

        # initialize weights(gamma) and bias(beta)
        self.initialize(None,None)

        self.alpha=0.8

        # for forward
        self.epsilon = 1e-11 # smaller than 1e-10
        self.test_mean = 0
        self.test_var = 1
        self.xhat=0
        self.mean=0
        self.var=0

        # for backward
        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None
        self._bias_optimizer = None

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, y):
        self._gradient_weights = y

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, xy):
        self._gradient_bias = xy

    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def reformat(self, input_tensor):
        return np.reshape(input_tensor, (-1, self.channels))
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        is_conv = input_tensor.ndim == 4
        self.is_conv=is_conv
        axis_map = {False: 0, True: (0, 2, 3)}  
        mean_ax = axis_map[is_conv]  
        axis_map1 = {False: 0, True: (0, 2, 3)}  
        var_ax = axis_map1[is_conv] 
        

        self.mean = np.mean(input_tensor, 
        axis=mean_ax)
        self.var = np.var(input_tensor, 
        axis=var_ax)

        if not self.is_conv:
            xhat_calculations = {
            True: lambda: (input_tensor - self.test_mean) / np.sqrt(self.test_var + self.epsilon),
            False: lambda: (self.input_tensor - self.mean) / np.sqrt(self.var + self.epsilon),}

            self.xhat = xhat_calculations[self.testing_phase]()

            if not self.testing_phase:
                self.test_mean = self.alpha * self.mean + (1 - self.alpha) * self.mean
                self.test_var = self.alpha * self.var + (1 - self.alpha) * self.var

            return self.weights * self.xhat + self.bias
        else:
            bsize = input_tensor.shape[0]
            channels = input_tensor.shape[1]
            if self.testing_phase:
                input_tensor_copy = self.input_tensor.copy() 
                input_tensor_copy -= self.test_mean.reshape((1, channels, 1, 1))
                input_tensor_copy /= np.sqrt(self.test_var.reshape((1, channels, 1, 1)) + self.epsilon)
                return input_tensor_copy
                
            new_mean = np.mean(self.input_tensor, axis=mean_ax)
            new_var = np.var(self.input_tensor, axis=var_ax)

            self.test_mean = self.alpha * self.mean[:, np.newaxis, np.newaxis] + (1 - self.alpha) * new_mean[:, np.newaxis, np.newaxis]
            self.test_var = self.alpha * self.var[:, np.newaxis, np.newaxis] + (1 - self.alpha) * new_var[:, np.newaxis, np.newaxis]

            self.mean = new_mean
            self.var = new_var
            
            self.xhat = (self.input_tensor - self.mean[:, np.newaxis, np.newaxis]) / np.sqrt(self.var[:, np.newaxis, np.newaxis] + self.epsilon)
           
            return self.weights[:, np.newaxis, np.newaxis] * self.xhat + self.bias[:, np.newaxis, np.newaxis]
    
    def backward(self, error_tensor):
        
        axis = (0, 2, 3) if self.is_conv else 0

        compute_and_reformat = lambda et, it: self.reformat(compute_bn_gradients(self.reformat(et), self.reformat(it),
                                                                        self.weights, self.mean, self.var, self.epsilon))

        error_here = compute_and_reformat(error_tensor, self.input_tensor) if self.is_conv else compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var, self.epsilon)

        gradients = {
        'weights': np.sum(error_tensor * self.xhat, axis=axis),
        'bias': np.sum(error_tensor, axis=axis)
        }
        self.gradient_weights = gradients['weights']
        self.gradient_bias = gradients['bias']

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return error_here

    def reformat(self, tensor):
        check_4d = lambda x: len(x.shape) == 4
        a2d_crr = check_4d(tensor)
        
        if a2d_crr:
            batch, ex, r, c = tensor.shape
            to = tensor.reshape(batch, ex, r * c).transpose(0, 2, 1).reshape(-1, ex)
        else:
            batch, ex, r, c = self.input_tensor.shape
            to = tensor.reshape(batch, r * c, ex).transpose(0, 2, 1).reshape(batch, ex, r, c)
        return to