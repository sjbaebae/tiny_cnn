import numpy as np
import models

class Conv2d(models.NN):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True, activation="relu"):
        super().__init__()
        self.weight = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)

        if bias:
            self.bias = np.random.rand(out_channels)
        else:
            self.bias = None

        