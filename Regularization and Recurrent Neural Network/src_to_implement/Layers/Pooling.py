import numpy as np
from Layers.Base import BaseLayer
from itertools import product
from scipy.ndimage import maximum_filter
class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        # Store the input tensor
        self.input_tensor = input_tensor

        # Get the dimensions of the input tensor
        batch_count, channel_count, input_height, input_width = [dim for dim in input_tensor.shape]

        # Get stride dimensions 
        stride_iterator = iter(self.stride_shape)
        stride_vertical = next(stride_iterator)
        stride_horizontal = next(stride_iterator)

        # Get pooling dimensions 
        pooling_iterator = iter(self.pooling_shape)
        pooling_height = next(pooling_iterator)
        pooling_width = next(pooling_iterator)

        # Calculate output dimensions (height and width)
        output_height = int((input_height - pooling_height) / stride_vertical + 1)
        output_width = int((input_width - pooling_width) / stride_horizontal + 1)

        # Initialize the output tensor with zeros
        output_result = np.empty((batch_count, channel_count, output_height, output_width))
        output_result.fill(0)

        # Perform max pooling operation
        for output_h_idx, output_w_idx in product(range(output_height), range(output_width)):
            pooled_area = input_tensor[:, :, output_h_idx * stride_vertical : output_h_idx * stride_vertical + pooling_height, output_w_idx * stride_horizontal : output_w_idx * stride_horizontal + pooling_width]

            # Find maximum value within the pooling window
            output_result[:, :, output_h_idx, output_w_idx] = np.max(pooled_area, axis=(2, 3))

        self.output = output_result
        return output_result


    def backward(self, error_tensor):
        # Create an empty gradient tensor to hold the result.
        input_tensor_gradient = np.empty_like(self.input_tensor)
        input_tensor_gradient.fill(0)  # Initialize with zeros

        # Extract dimensions from input_tensor shape.
        num_batches, num_channels, input_height, input_width = self.input_tensor.shape

        # Get stride and pooling dimensions.
        stride_iterator = iter(self.stride_shape)
        stride_height = next(stride_iterator)
        stride_width = next(stride_iterator)
        pool_size_iterator = iter(self.pooling_shape)
        pool_height = next(pool_size_iterator)
        pool_width = next(pool_size_iterator)

        # Get output height and width from self.output.
        _, _, output_height, output_width = self.output.shape

        # Iterate over batches, channels, and output dimensions.
        for batch_idx, channel_idx, out_h, out_w in product(
            range(num_batches), range(num_channels), range(output_height), range(output_width)
        ):
            # Get the indices of the maximum value within the pooling window.
            # Find the flattened index of the maximum value
            flattened_max_index = np.argmax(
                self.input_tensor[
                    batch_idx,
                    channel_idx,
                    out_h * stride_height : out_h * stride_height + pool_height,
                    out_w * stride_width : out_w * stride_width + pool_width,
                ]
            )

            # Convert the flattened index to 2D coordinates
            max_height_index = flattened_max_index // pool_width
            max_width_index = flattened_max_index % pool_width

            # Update gradient tensor with error_tensor value at the max index.
            np.add.at(
                input_tensor_gradient[batch_idx, channel_idx],
                (out_h * stride_height + max_height_index, out_w * stride_width + max_width_index),
                error_tensor[batch_idx, channel_idx, out_h, out_w],
            )

        return input_tensor_gradient