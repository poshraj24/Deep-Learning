import numpy as np
import math

class Sgd:
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor += -self.learning_rate*gradient_tensor #updates the weight using GD
        return weight_tensor
    
class  SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate:float):
        self.learning_rate=learning_rate
        self.momentum_rate=momentum_rate
        self.previous_gradient=0

    def calculate_update(self,weight_tensor, gradient_tensor):
        self.previous_gradient = self.previous_gradient * self.momentum_rate - self.learning_rate * gradient_tensor
        return self.previous_gradient + weight_tensor

class Adam():
    def __init__(self, learning_rate: float, mu:float, rho: float):
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
        return weight_tensor - self.learning_rate * gradient_tensor