from .main import Node, Edge
import numpy as np
from ..activations.functions import softmax

class ReluBackward(Node):
    def __init__(self, left):
        super().__init__(edges=(Edge(left.grad_fn, 0),), saved_tensors=(left,))
    
    def apply(self, grad_in: np.ndarray):
        return grad_in * (self.saved_tensors[0].data > 0)

class GeluBackward(Node):
    def __init__(self, left):
        super().__init__(edges=(Edge(left.grad_fn, 0),), saved_tensors=(left,))

    
    def apply(self, grad_in: np.ndarray):
        # original gelu is: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) or 0.5x + 0.5x * tanh(...) * (x + 0.044715x^3)
        # then derivative becomes: 0.5 + 0.5x^2/x * tan() + 0.5x * (sec^2(...)) + 
        x = self.saved_tensors[0].data
        u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        du_dx = np.sqrt(2 / np.pi) * (1 + 0.044715 * 3 * x**2)
        local_grad = 0.5 * (1 + np.tanh(u) + x * (1 - np.tanh(u)**2) * du_dx)
        return grad_in * local_grad

class SiluBackward(Node):
    def __init__(self, left):
        super().__init__(edges=(Edge(left.grad_fn, 0),), saved_tensors=(left,))
    
    def apply(self, grad_in: np.ndarray):
        # y = x * (1 / (1 + np.exp(-x)))
        x = self.saved_tensors[0].data
        dy_dx = 1 / (1 + np.exp(-x)) + x * np.exp(-x) / (1 + np.exp(-x))**2

        return grad_in * dy_dx

class SigmoidBackward(Node):
    def __init__(self, left):
        super().__init__(edges=(Edge(left.grad_fn, 0),), saved_tensors=(left,))
    
    def apply(self, grad_in: np.ndarray):
        # y = 1 / (1 + np.exp(-x))
        x = self.saved_tensors[0].data
        dy_dx = np.exp(-x) / (1 + np.exp(-x))**2
        return grad_in * dy_dx

class TanhBackward(Node):
    def __init__(self, left):
        super().__init__(edges=(Edge(left.grad_fn, 0),), saved_tensors=(left,))
    
    def apply(self, grad_in: np.ndarray):
        # y = tanh(x)
        x = self.saved_tensors[0].data
        dy_dx = 1 - np.tanh(x)**2
        return grad_in * dy_dx
        
class SoftmaxBackward(Node):
    def __init__(self, tensor):
        super().__init__(edges=(Edge(tensor.grad_fn, 0),), saved_tensors=(tensor,))
    
    def apply(self, grad_in: np.ndarray):
        # dont build full matrix. instead, use the fact that dy_dx = y * (1 - y), where y can reuse the softmax function
        x = self.saved_tensors[0].data
        y = softmax(x)
        dot_prod = np.sum(grad_in * y, axis = 1, keepdims = True)
        return y * (grad_in - dot_prod)
        
        
        