import numpy as np

class L1_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weight_tensor):
        return self.alpha * np.sign(weight_tensor)
    
    def norm(self, weight_tensor):
        return self.alpha * np.sum(np.abs(weight_tensor))
    
class L2_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weight_tensor):
        return 2 * self.alpha * weight_tensor

    def norm(self, weight_tensor):
        return self.alpha * np.sum(weight_tensor ** 2)