import numpy as np
import math

class Optimizer:
    def __init__(self, learning_rate= None):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor-self.learning_rate*gradient_tensor
        if self.regularizer is not None:
            updated_weight = updated_weight - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        return updated_weight
    

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate:float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate=momentum_rate
        self.prev_gradient = 0


    def calculate_update(self, weight_tensor, gradient_tensor):
        self.prev_gradient = self.prev_gradient * self.momentum_rate - self.learning_rate * gradient_tensor
        updated_weight = self.prev_gradient + weight_tensor - (
            self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) if self.regularizer is not None else 0
        )
        return updated_weight
    
class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu:float, rho: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu=mu
        self.rho=rho

        self.moving_average_of_grad = 0
        self.squared_moving_average_of_grad=0
        self.counter_var =1


    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.moving_average_of_grad = gk + self.mu * (self.moving_average_of_grad - gk)
        squared_grad_update = (1 - self.rho) * np.square(gk) 
        # Update the squared moving average
        self.squared_moving_average_of_grad = self.rho * self.squared_moving_average_of_grad + squared_grad_update
        rcap = self.squared_moving_average_of_grad / (1 - np.power(self.rho, self.counter_var))
        vcap = self.moving_average_of_grad / (1 - self.mu ** self.counter_var)
        self.counter_var += 1
        eps = math.ulp(1.0)
        denominator = np.sqrt(rcap) + eps
        gradient_tensor = vcap / denominator
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor

        if self.regularizer is not None:
            reg_gradient = self.regularizer.calculate_gradient(weight_tensor)
            updated_weight -= self.learning_rate * reg_gradient
            
        return updated_weight