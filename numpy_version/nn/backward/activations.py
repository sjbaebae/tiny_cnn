from .main import Node, Edge, Function
import numpy as np
from ..activations.functions import softmax, relu, gelu, silu, sigmoid, tanh
from .core import get_edge

class ReluBackward(Function):
    @staticmethod
    def forward(data):
        return relu(data)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
    
    def backward(self, grad_in: np.ndarray):
        return grad_in * (getattr(self.saved_tensors[0], "data", self.saved_tensors[0]) > 0),

class GeluBackward(Function):
    @staticmethod
    def forward(data):
        return gelu(data)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
    
    def backward(self, grad_in: np.ndarray):
        x = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        du_dx = np.sqrt(2 / np.pi) * (1 + 0.044715 * 3 * x**2)
        local_grad = 0.5 * (1 + np.tanh(u) + x * (1 - np.tanh(u)**2) * du_dx)
        return grad_in * local_grad,

class SiluBackward(Function):
    @staticmethod
    def forward(data):
        return silu(data)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
    
    def backward(self, grad_in: np.ndarray):
        x = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        dy_dx = 1 / (1 + np.exp(-x)) + x * np.exp(-x) / (1 + np.exp(-x))**2
        return grad_in * dy_dx,

class SigmoidBackward(Function):
    @staticmethod
    def forward(data):
        return sigmoid(data)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
    
    def backward(self, grad_in: np.ndarray):
        x = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        dy_dx = np.exp(-x) / (1 + np.exp(-x))**2
        return grad_in * dy_dx,

class TanhBackward(Function):
    @staticmethod
    def forward(data):
        return tanh(data)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
    
    def backward(self, grad_in: np.ndarray):
        x = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        dy_dx = 1 - np.tanh(x)**2
        return grad_in * dy_dx,
        
class SoftmaxBackward(Function):
    @staticmethod
    def forward(data):
        return softmax(data)

    def __init__(self, tensor):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=(tensor,))
    
    def backward(self, grad_in: np.ndarray):
        x = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        y = softmax(x)
        dot_prod = np.sum(grad_in * y, axis = 1, keepdims = True)
        return y * (grad_in - dot_prod),