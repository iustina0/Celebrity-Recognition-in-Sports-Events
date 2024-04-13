import numpy as np
from layer import Layer
from scipy import signal

class Convlution(Layer):
    def __init__(self, input_shape, kernle_size, depth):
        input_depth, input_height, input_wight  = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernle_size + 1, input_wight - kernle_size + 1)
        self.kernles_shape = (depth, input_depth, kernle_size, kernle_size)
        self.kernles = np.random.randn(*self.kernles_shape)
        self.biases = np.random.randn(*self.output_shape)

    def foward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernles[i, j], "valid")
        return self.output
    
    def backwards(self, output_gradient, lerning_rate):
        kernle_gradient = np.zeros(self.input_depth)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernle_gradient[i, j] = signal.correlate2d(self.input[1], output_gradient, "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernles[i, j], "full")

        self.kernles -= lerning_rate * kernle_gradient
        self.biases -= lerning_rate * output_gradient
        return input_gradient