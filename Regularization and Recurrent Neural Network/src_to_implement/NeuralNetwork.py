from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer


    
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, value):
        self._phase = value
    def forward(self):
        # Retrieve input and output from the data layer
        self.input_data, self.label = self.data_layer.next()
        
        regularization_loss = 0  # Initialize regularization loss
        j = 0
        
        # Forward pass through each layer
        while j < len(self.layers):
            self.input_data = self.layers[j].forward(self.input_data)
            try:
                regularization_loss += self.optimizer.regularizer.norm(self.layers[j].weights)
            except:
                pass        
            self.layers[j].testing_phase = True  # Set testing phase to True
            j += 1
            
        # Forward pass through the loss layer with added regularization loss
        self.pred = self.loss_layer.forward(self.input_data + regularization_loss, self.label)
        return self.pred
    
    def backward(self):
        loss = self.loss_layer.backward(self.label)
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
