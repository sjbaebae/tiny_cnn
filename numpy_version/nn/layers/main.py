from numpy_version.nn.backward.activations import ReluBackward, GeluBackward, SiluBackward, SigmoidBackward, TanhBackward
import numpy as np
from ...tensor import Tensor
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
        # backward function
        out = Tensor(relu(x.data), requires_grad = x.requires_grad)
        out.grad_fn = ReluBackward(x)
        return out

class Gelu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        # backward function
        out = Tensor(gelu(x.data), requires_grad = x.requires_grad)
        out.grad_fn = GeluBackward(x)
        return out

class Silu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        # backward function
        out = Tensor(silu(x.data), requires_grad = x.requires_grad)
        out.grad_fn = SiluBackward(x)
        return out

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        # backward function
        out = Tensor(sigmoid(x.data), requires_grad = x.requires_grad)
        out.grad_fn = SigmoidBackward(x)
        return out

class Tanh(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        # backward function
        out = Tensor(tanh(x.data), requires_grad = x.requires_grad)
        out.grad_fn = TanhBackward(x)
        return out

# TAYLOR SERIES EXPANSION -> SUM(f'(x) * (x - x_0)^n / n!)