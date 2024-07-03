import copy
import numpy as np


from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.regularization_loss = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.hidden_fcl = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.output_fcl = FullyConnected(self.hidden_size, self.output_size)

        self.hidden_fcl_input_tensor = np.ndarray([])
        self.output_fcl_input_tensors = []
        self.sigmoid_outputs = []
        self.tanh_outputs = []

        self.hidden_fcl_gradient_weights = []
        self.output_fcl_gradient_weights = []
        
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.hidden_fcl.weights

    @weights.setter
    def weights(self, value):
        self.hidden_fcl.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(self.hidden_fcl.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.output_fcl.weights)
        return self.regularization_loss
        

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fcl.initialize(weights_initializer, bias_initializer)
        self.output_fcl.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.sigmoid_outputs = []
        self.tanh_outputs = []
        self.output_fc_input_tensors = []
        self.hidden_fcl_input_tensors = []
        previous_hstate = self.hidden_state.copy() if self.memorize else np.zeros(self.hidden_size)
        batch_size = input_tensor.shape[0]
        output_tensor = np.zeros((batch_size,self.output_size))

        # for each batch dimension or time dimension?
        for i, inp_r in enumerate(input_tensor):
            # for first step, input will be input_tensor and 0
            # for next steps, input will be input_tensor and previous hidden state
            # use the hidden state as input to the hidden layer
            n_inp = np.concatenate([[previous_hstate], [inp_r]], axis=1)

            curr_hstate = self.tanh.forward(self.hidden_fcl.forward(n_inp))
            previous_hstate = curr_hstate[0]

            output_tensor[i] = self.sigmoid.forward(self.output_fcl.forward(curr_hstate))[0]

            # store inputs and outputs for backprop
            self.hidden_fcl_input_tensors += [self.hidden_fcl.input_tensor]
            self.output_fc_input_tensors += [self.output_fcl.input_tensor]
            self.sigmoid_outputs += [self.sigmoid.out]
            self.tanh_outputs += [self.tanh.out]

        # update hidden state
        self.hidden_state = curr_hstate[0]

        return output_tensor

    def backward(self, error_tensor):
        # Initialize gradients for weights of hidden and output fully connected layers
        self.gradient_weights = np.zeros(self.hidden_fcl.weights.shape)
        self.output_fcl_gradient_weights = np.zeros(self.output_fcl.weights.shape)
        
        hstate_prev_grad = 0
        batch_size = error_tensor.shape[0]
        gradient_wrt_inputs = np.zeros((batch_size, self.input_size))

        # Iterate through each time step in reverse order
        for step in reversed(range(error_tensor.shape[0])):
            # Retrieve the sigmoid output at the current time step
            self.sigmoid.out = self.sigmoid_outputs[step]
            sigmoid_error = self.sigmoid.backward(error_tensor[step])

            # Retrieve the input to the output layer at the current time step and compute the error
            temp_input_tensor = self.output_fc_input_tensors[step]
            self.output_fcl.input_tensor = temp_input_tensor
            output_fcl_error = self.output_fcl.backward(sigmoid_error)

            # Retrieve the tanh output at the current time step
            self.tanh.out = self.tanh_outputs[step]
            tanh_error = self.tanh.backward(output_fcl_error + hstate_prev_grad)

            # Pass the error in tanh to the hidden layer
            self.hidden_fcl.input_tensor = self.hidden_fcl_input_tensors[step]
            hidden_fcl_error = self.hidden_fcl.backward(tanh_error)

            # Split the error in the hidden layer into two parts: hidden state and input
            hstate_prev_grad, _ = np.split(hidden_fcl_error, [self.hidden_size], axis=1)
            gradient_with_respect_to_input = np.array_split(hidden_fcl_error, [self.hidden_size], axis=1)[1]
            gradient_wrt_inputs[step] = gradient_with_respect_to_input[0]

            # Accumulate the gradient weights for hidden and output fully connected layers
            self.gradient_weights = np.add(self.gradient_weights, self.hidden_fcl.gradient_weights)
            self.output_fcl_gradient_weights = np.add(self.output_fcl_gradient_weights, self.output_fcl.gradient_weights)

        # Update the weights using the accumulated gradients
        if self.optimizer:
            self.output_fcl.weights = self.optimizer.calculate_update(
                self.output_fcl.weights, self.output_fcl_gradient_weights
            )
            self.hidden_fcl.weights = self.optimizer.calculate_update(
                self.hidden_fcl.weights, self.gradient_weights
            )

        return gradient_wrt_inputs




    