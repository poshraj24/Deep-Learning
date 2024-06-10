import numpy as np 

class Constant():
    def __init__(self,value_constant=0.1):
        self.value_constant=value_constant
    def initialize(self, weights_shape, fan_in, fan_out):
        self.init_tensor = np.ones(weights_shape) * self.value_constant
        return self.init_tensor

class UniformRandom():
    #Initializes weights randomly from a uniform distribution
    def initialize(self,weights_shape, fan_in, fan_out):
        rng = np.random.default_rng()
        self.init_tensor = rng.uniform(size=(weights_shape[0], weights_shape[1]))
        return self.init_tensor

class Xavier():
    def initialize(self,weights_shape, fan_in, fan_out):
        scale = (2 / (fan_in + fan_out))**0.5
        rng = np.random.default_rng()
        self.init_tensor = scale * rng.standard_normal(weights_shape)
        return self.init_tensor

class He():
    def initialize(self,weights_shape, fan_in, fan_out):
        scale = (2 / fan_in)**0.5
        rng = np.random.default_rng()
        self.init_tensor = scale * rng.standard_normal(weights_shape)
        return self.init_tensor

