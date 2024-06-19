import numpy as np
from Layers.Base import *
from Layers.Helpers import *
from functools import partial

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gamma = np.ones(channels, dtype=float)
        self.beta = np.zeros(channels, dtype=float)
        self.epsilon = 1e-8
        self.trainable = True
        self.testing_phase = False
        self.mean = None
        self.var = None
        self.optimizer = None
        self.gradient_tensor = None
        
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

    def initialize(self, weights_initializer, bias_initializer):
        self.weights= np.ones(self.channels, dtype=float)
        self.bias = np.zeros(self.channels, dtype=float)

    def reformat(self, input_tensor):
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
            return input_tensor.reshape(N, C*H*W)
        else:
            return input_tensor


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.is_conv = len(input_tensor.shape) == 4
        axes = {True: (0, 2, 3), False: 0}
        mean_ax = axes[self.is_conv]
        var_ax = axes[self.is_conv]
        mean_func = lambda x, axis: np.mean(x, axis=axis)
        var_func = lambda x, axis: np.var(x, axis=axis)
        self.mean = mean_func(input_tensor, mean_ax)
        self.var = var_func(input_tensor, var_ax)

        if self.is_conv == False:
            self.xhat = (
            (input_tensor - self.test_mean) / np.sqrt(self.test_var + self.epsilon)
            if self.testing_phase
            else (self.input_tensor - self.mean) / np.sqrt(self.var + self.epsilon))
       
            if not self.testing_phase:
                self.test_mean = self.alpha * self.mean + (1 - self.alpha) * self.mean
                self.test_var = self.alpha * self.var + (1 - self.alpha) * self.var
            return self.weights*self.xhat + self.bias
        else:
            bsize,channels,*_ = input_tensor.shape
            if self.testing_phase:
                inv_stddev = np.reciprocal(np.sqrt(self.test_var.reshape((1, channels, 1, 1)) + self.epsilon))
                return (self.input_tensor - self.test_mean.reshape((1, channels, 1, 1))) * inv_stddev
            new_mean = np.average(self.input_tensor, axis=mean_ax)  # Equivalent to np.mean
            new_var =  np.nanvar(self.input_tensor, axis=var_ax)  # Handles NaNs (if relevant)
            reshaped_mean = self.mean.reshape((1, channels, 1, 1))
            reshaped_var = self.var.reshape((1, channels, 1, 1))

            self.test_mean = self.alpha * reshaped_mean + (1 - self.alpha) * new_mean.reshape(
                (1, channels, 1, 1)
            )
            self.test_var = self.alpha * reshaped_var + (1 - self.alpha) * new_var.reshape(
                (1, channels, 1, 1)
            )
            self.mean = new_mean
            self.var = new_var

            reshaped_mean = self.mean.reshape((1, channels, 1, 1))
            reshaped_var = self.var.reshape((1, channels, 1, 1))

            self.xhat = (self.input_tensor - reshaped_mean) / np.sqrt(reshaped_var + self.epsilon)
            self.weights.shape = (1, channels, 1, 1)  
            self.bias.shape = (1, channels, 1, 1)

            return self.weights * self.xhat + self.bias
    
    def backward(self, error_tensor):
        axis = not self.is_conv * (0,) or self.is_conv * (0, 2, 3) 
        if self.is_conv: 
            compute_bn_grads_with_params = partial(compute_bn_gradients, weights=self.weights, mean=self.mean, var=self.var, epsilon=self.epsilon)
            

            err_here = compute_bn_grads_with_params(self.reformat(error_tensor), self.reformat(self.input_tensor))
            err_here = self.reformat(err_here)
        else:
            err_here = compute_bn_gradients(error_tensor, self.input_tensor, self.mean, self.var, self.epsilon, self.weights)
        self.gradient_weights = np.tensordot(error_tensor, self.xhat, axes=(axis, axis))
        self.gradient_bias = np.tensordot(error_tensor, np.ones_like(error_tensor), axes=(axis, axis))

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return err_here
    
    def reformat(self,tensor):
        make_2d = [False, True][len(tensor.shape) == 4]
        if make_2d:
            batch, ex, r, c = tensor.shape
            out = tensor.reshape(batch, ex, r * c).transpose(0, 2, 1).reshape(-1, ex)
        else:
            batch, ex, r, c = self.input_tensor.shape
            out = tensor.reshape((batch, r * c, ex)).transpose(0, 2, 1).reshape(batch, ex, r, c)  


        return out


                 

        