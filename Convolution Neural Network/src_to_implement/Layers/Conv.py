from Layers.Base import BaseLayer
from itertools import product
import numpy as np
from scipy import signal
from functools import reduce
import operator
from copy import deepcopy as copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = self._get_stride_shape(stride_shape)
        self.convolution_shape = convolution_shape #shape of conv kernel
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.rand(num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._bias_optimizer = None
        self.conv_dim = 2 if len(convolution_shape) == 3 else 1

    def _get_stride_shape(self, stride_shape):
        return (stride_shape[0], stride_shape[0]) if len(stride_shape) == 1 else stride_shape


    def initialize(self, weights_initializer, bias_initializer):
        self._initialize_weights(weights_initializer)
        self._initialize_bias(bias_initializer)
        self._initialize_optimizers()

    def _initialize_weights(self, weights_initializer):
        self.weights = weights_initializer.initialize(
            self.weights.shape,
            reduce(operator.mul, self.convolution_shape),
            reduce(operator.mul, [self.num_kernels, *self.convolution_shape[1:]])
        )

    def _initialize_bias(self, bias_initializer):
        self.bias = bias_initializer.initialize(
            self.bias.shape, 1, self.num_kernels
        )

    def _initialize_optimizers(self):
        self._optimizer = copy(self.optimizer)
        self._bias_optimizer = copy(self.optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        ishape = input_tensor.shape
        self.ishape = ishape
        
        # Unpack batch_size and channel count
        bsize, c = ishape[:2]  

        #height and width of input tensor for 2d convolution
        y, x = ishape[2:] if self.conv_dim == 2 else (ishape[2], None)
        
        #kernel height and width
        cx, cy = self.convolution_shape[1], self.convolution_shape[2] if self.conv_dim == 2 else (self.convolution_shape[1], None)
        cx, cy = self.convolution_shape[1], self.convolution_shape[2] if self.conv_dim == 2 else (self.convolution_shape[1], None)
        
        #strides for height and width
        sh, sw = self.stride_shape if self.conv_dim == 2 else (self.stride_shape[0], None)

        #padding
        pad = np.floor_divide([cx-1],2).tolist()
        out_shape = []        # (After convolution) Start with an empty list
        out_shape.append(int((y - cx + 2 * pad[0]) / sh) + 1) 
        if self.conv_dim==2:
            pad.append((cy-1)/2)
            out_shape.append(int((x-cy+2*pad[1])/sw)+1)
        self.pad=pad

        result = np.empty((bsize, self.num_kernels, *out_shape))
        result[:] = 0

        # if used correlation in forward, should use convolve in backward 
        for cb, ck in product(range(bsize), range(self.num_kernels)):
                # conv filter 
                kout = np.sum(
                    [signal.correlate(input_tensor[cb, ch], self.weights[ck, ch], mode='same', method='direct') for ch in range(c)],
                    axis=0
                )

                # Apply striding
                
                kout = kout[::sh] if self.conv_dim != 2 else kout[::sh, ::sw]

                # Add bias
                result[cb, ck] = kout + self.bias[ck].reshape((1, 1) if self.conv_dim == 2 else (1,))

        return result

    def update_parameters(self, error_tensor):
        #here error_tensor stores error tensor from backward pass

        yerror = np.sum(np.sum(error_tensor, axis=0), axis=1)
                            
        # compute gradient for the bias
        self._gradient_bias = np.sum(np.expand_dims(yerror, axis=1) if self.conv_dim == 1 else yerror, axis=1) 

                
        batch_size, channels = self.ishape[:2]  # Extract batch and channel dimensions

        if self.conv_dim == 2:  # 2D case
            height, width = self.ishape[2:] 
            spatial_dims = (height, width)
        else:  # 1D case
            height = self.ishape[2]
            spatial_dims = (height,)
        # Unpack height (or length)
        y = spatial_dims[0] 
        # Unpack width (or set to None for 1D)
        x = spatial_dims[1] if self.conv_dim == 2 else None  

        
        try:
            sh, sw = self.stride_shape  # Try to unpack both stride values
        except ValueError:  # If 1D, there's only one value
            sh = self.stride_shape[0]
            sw = 1  # Default stride for width in 1D

        # Kernel dimensions
        cx = self.convolution_shape[-2]  # Unpack kernel height
        cy = self.convolution_shape[-1:]  # Get the last element as a list
        # If 1D, ensure cy is a single integer value
        cy = cy[0] if cy else 1 

        self.gradient_weights=np.array([np.zeros(w.shape) for w in self.weights])
        # Precompute padding widths and input tensor padding for all cases
        for cb, ch, ck in product(range(batch_size), range(channels), range(self.num_kernels)):
            if self.conv_dim==2:
                        error = np.empty((y, x))
                        error.fill(0)
                        error_reshaped = error.reshape(-1)
                        for i in range(0, error.shape[0], sh):
                            for j in range(0, error.shape[1], sw):
                                error[i, j] = error_tensor[cb, ck, i // sh, j // sw]
                        # Create padding for input tensor
                        pad_width = [(int(np.ceil(self.pad[i])), int(np.floor(self.pad[i]))) for i in range(self.conv_dim)]

                        # Pad input tensor along spatial dimensions
                        inp = np.pad(self.input_tensor[cb, ch], pad_width) 
            else:
                error = np.zeros(y)
                for i in range(0, error.shape[0], sh):
                    error[i] = error_tensor[cb, ck, i // sh]  
                            # Extract the input slice for the current batch and channel
                inp_slice = self.input_tensor[cb, ch]

                # Calculate padding values
                pad_before = int(np.ceil(self.pad[0]))
                pad_after = int(np.floor(self.pad[0]))

                # Create padding arrays and concatenate
                zeros_before = np.zeros(pad_before)
                zeros_after = np.zeros(pad_after)
                inp = np.concatenate([zeros_before, inp_slice, zeros_after]) # Pad input tensor along spatial dimensions


            self.gradient_weights[ck, ch] += signal.correlate(inp, error, mode='valid')

        if self.optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

    def error_this_layer(self, error_tensor):
        gradient = np.zeros(self.input_tensor.shape, dtype=self.input_tensor.dtype)
        sh, sw = self.stride_shape[0], self.stride_shape[1] if len(self.stride_shape) > 1 else 1
       
        nweight = self.weights.copy()
        # Dynamically determine the axes to transpose
        transpose_axes = (1, 0) + tuple(range(2, 2 + self.conv_dim))  # (1, 0, 2, 3) for 2D, (1, 0, 2) for 1D

        # Transpose weights 
        nweight = np.transpose(nweight, axes=transpose_axes)
        ishape = self.input_tensor.shape

        #dictionary for dimensions based on convolution type
        dims = {
            2: ishape[-2:],
            'default': (ishape[-1], None)
        }
        y, x = dims.get(self.conv_dim, dims['default'])

        bsize = self.input_tensor.shape[0]
        wk, wc = map(lambda i: nweight.shape[i], range(2))
        for cb, ck in product(range(bsize), range(wk)):
                grad = 0
                for c in range(wc):
                    if self.conv_dim == 2:
                        err = np.zeros((y, x))
                        err[np.arange(0, y, sh)[:, None], np.arange(0, x, sw)] = error_tensor[cb, c]
                    else:
                        err = np.zeros(y)
                        err[np.arange(0, y, sh)] = error_tensor[cb, ck]
                    # we used correlate on forward, use convolve now
                    grad = np.add(grad, signal.convolve(err, nweight[ck, c], mode='same', method='direct'))
                    
                gradient[cb, ck] = grad

        return gradient
        

    def backward(self, error_tensor):
        self.update_parameters(error_tensor)
        gradient = self.error_this_layer(error_tensor)
        
        return gradient

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
