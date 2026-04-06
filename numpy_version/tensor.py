import numpy as np
import nn
from nn.backward.core import AddBackward, SubBackward, MulBackward, DivBackward, PowBackward, NegBackward, AbsBackward, MatmulBackward, SliceBackward, PermuteBackward, ViewBackward

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, grad_fn = None):
        self.data = data
        self.requires_grad = requires_grad
        if self.is_leaf:
            self.grad = np.zeros_like(data)
        self.grad_fn = grad_fn

    def backward(self, grad_in: np.ndarray = np.array([1])):
        self.grad_fn(grad_in)
    
    def __add__(self, other):
        raw_data = self.data + other.data
        result = Tensor(raw_data, requires_grad = self.requires_grad or other.requires_grad, grad_fn = nn.backward.core.AddBackward(self, other))
        return result
    
    def __sub__(self, other):
        raw_data = self.data - other.data
        result = Tensor(raw_data, requires_grad = self.requires_grad or other.requires_grad, grad_fn = SubBackward(self, other))
        return result
    
    def __mul__(self, other):
        result = Tensor(self.data * other.data, requires_grad = self.requires_grad or other.requires_grad)
        result.grad_fn = MulBackward(self, other)
        return result
    
    def __truediv__(self, other):
        result = Tensor(self.data / other.data, requires_grad = self.requires_grad or other.requires_grad)
        result.grad_fn = DivBackward(self, other)
        return result
    
    def __pow__(self, other):
        result = Tensor(self.data ** other.data, requires_grad = self.requires_grad or other.requires_grad)
        result.grad_fn = PowBackward(self, other)
        return result
    
    def __neg__(self):
        result = Tensor(-self.data, requires_grad = self.requires_grad)
        result.grad_fn = NegBackward(self)
        return result
    
    def __abs__(self):
        result = Tensor(np.abs(self.data), requires_grad = self.requires_grad)
        result.grad_fn = AbsBackward(self)
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return Tensor(self.data[key])
    
    def __setitem__(self, key, value):
        self.data[key] = value

    # SET ATTRIBUTE DOES NOT EXIST. NO NEED FOR TENSORS
    
    def __getattr__(self, name):
        # forward to numpy data
        if self.requires_grad:
            # how do we do grad flow? block for now
            raise AttributeError(f"Cannot access attribute {name} on Tensor with requires_grad=True")
        else:
            return self.data.__getattribute__(name)

    def __matmul__(self, other):
        result = Tensor(self.data @ other.data, requires_grad = self.requires_grad or other.requires_grad)
        result.grad_fn = MatmulBackward(self, other)
        return result
    
    def reshape(self, shape: tuple): 
        self.data.reshape(shape)

    def permute(self, dims: tuple):
        result = Tensor(self.data.transpose(dims), requires_grad = self.requires_grad)
        result.grad_fn = PermuteBackward(self, dims)
        return result

    def view(self, shape: tuple):
        result = Tensor(self.data.reshape(shape), requires_grad = self.requires_grad)
        result.grad_fn = ViewBackward(self, shape)
        return result

    def slice(self, key):
        result = Tensor(self.data[key], requires_grad = self.requires_grad)
        result.grad_fn = SliceBackward(self, key)
        return result

    @property
    def train(self):
        self.training = True
    
    @property
    def eval(self):
        self.training = False

    @property
    def is_leaf(self):
        return self.requires_grad and self.grad_fn is None

# Parameter (autotrack weights)
class Parameter:
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.requires_grad = requires_grad
        
