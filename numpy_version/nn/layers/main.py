from ..backward.activations import ReluBackward, GeluBackward, SiluBackward, SigmoidBackward, TanhBackward
import numpy as np
from tensor import Tensor
from ..activations.functions import relu, gelu, silu, sigmoid, tanh

class Module:
    def __init__(self):
        pass
    def forward(self, x: Tensor):
        pass
    def __call__(self, x: Tensor):
        return self.forward(x)
    def train(self):
        self.training = True
    def eval(self):
        self.training = False

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        # kaiming normalization
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2 / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def forward(self, x: Tensor):
        return x @ self.weight + self.bias

class Activation(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        pass

#############################
#          CLASSES          #
#############################

class Relu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        return ReluBackward.apply(x)

class Gelu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        return GeluBackward.apply(x)

class Silu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        return SiluBackward.apply(x)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        return SigmoidBackward.apply(x)

class Tanh(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        return TanhBackward.apply(x)

# TAYLOR SERIES EXPANSION -> SUM(f'(x) * (x - x_0)^n / n!)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1):
        super().__init__()
        # conv2d layers
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / (in_channels * kernel_size * kernel_size)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

    #  need a vectorized conv2d fast from cols

    # for this will need a helper function to get all the indices I want then prepare from extraction numpy
    
    def forward(self, x: Tensor):
        # forward with conv2d fast.
        out = Tensor(conv2d(x.data, self.weight.data, self.bias.data, self.stride, self.padding, self.dilation), requires_grad = x.requires_grad)
        out.grad_fn = Conv2dBackward(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return out
            
        
        