from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer=weights_initializer    #refactored 
        self.bias_initializer=bias_initializer          #refactored
        

    def forward(self):
        self.input_data, self.i = self.data_layer.next()
        j = 0
        while j < len(self.layers):
            self.input_data = self.layers[j].forward(self.input_data)
            j += 1
        self.prediction = self.loss_layer.forward(self.input_data, self.i)
        return self.prediction
    def backward(self):
        loss = self.loss_layer.backward(self.i)
        for j in reversed(self.layers):
            loss = j.backward(loss)
        return loss
    
    def append_layer(self, j):
        if j.trainable:
            j.optimizer = deepcopy(self.optimizer)
            j.initialize(self.weights_initializer,self.bias_initializer) #initializes trainable layers with the stored initializers
        self.layers += [j]

    def train(self, steps):
        self.loss = []
        for i in range(steps):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        test_result = input_tensor

        for j in self.layers:
            test_result = j.forward(test_result)

        return test_result