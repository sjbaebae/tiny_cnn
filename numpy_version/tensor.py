import numpy as np
import nn
from nn.backward.core import AddBackward, SubBackward, MulBackward, DivBackward, PowBackward, NegBackward, AbsBackward, MatmulBackward, SliceBackward, PermuteBackward, ViewBackward, ReshapeBackward

# torch style no_grad guard
_grad_enabled = True

class no_grad:
    global _grad_enabled
    def __enter__(self):
        self._grad_enabled = _grad_enabled # what was it before entering this?
        _grad_enabled = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _grad_enabled = self._grad_enabled


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, grad_fn = None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
    
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

    def __matmul__(self, other):
        result = Tensor(self.data @ other.data, requires_grad = self.requires_grad or other.requires_grad)
        result.grad_fn = MatmulBackward(self, other)
        return result
    
    def reshape(self, shape: tuple): 
        result = Tensor(self.data.reshape(shape), requires_grad = self.requires_grad)
        # pass original tensor so we can use as reference
        result.grad_fn = ReshapeBackward(self, self)
        return result

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

    # main function for engine
    def backward(self):
        engine = nn.Engine()
        engine.backward(self)

    @property
    def is_leaf(self):
        return self.requires_grad and self.grad_fn is None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def T(self):
        return Tensor(self.data.T, requires_grad = self.requires_grad)

# Parameter (autotrack weights)
class Parameter(Tensor):
    def __init__(self, data: Tensor):
        super().__init__(data.data, requires_grad=True)
        
        
        
