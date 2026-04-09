import numpy as np
from nn.backward.main import no_grad

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, grad_fn = None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
    
    def __add__(self, other):
        from nn.backward.core import AddBackward
        return AddBackward.apply(self, other)
    
    def __sub__(self, other):
        from nn.backward.core import SubBackward
        return SubBackward.apply(self, other)
    
    def __mul__(self, other):
        from nn.backward.core import MulBackward
        return MulBackward.apply(self, other)
    
    def __truediv__(self, other):
        from nn.backward.core import DivBackward
        return DivBackward.apply(self, other)
    
    def __pow__(self, other):
        from nn.backward.core import PowBackward
        return PowBackward.apply(self, other)
    
    def __neg__(self):
        from nn.backward.core import NegBackward
        return NegBackward.apply(self)
    
    def __abs__(self):
        from nn.backward.core import AbsBackward
        return AbsBackward.apply(self)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return Tensor(self.data[key])
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __matmul__(self, other):
        from nn.backward.core import MatmulBackward
        return MatmulBackward.apply(self, other)
    
    def reshape(self, *shape): 
        from nn.backward.core import ReshapeBackward
        return ReshapeBackward.apply(self, shape)

    def permute(self, *dims):
        from nn.backward.core import PermuteBackward
        return PermuteBackward.apply(self, dims)

    def view(self, shape: tuple):
        from nn.backward.core import ViewBackward
        return ViewBackward.apply(self, shape)

    def slice(self, key):
        from nn.backward.core import SliceBackward
        return SliceBackward.apply(self, key)
    
    def sum(self, axis=None, keepdims=False):
        from nn.backward.core import SumBackward
        return SumBackward.apply(self, axis=axis, keepdims=keepdims)

    def log(self):
        from nn.backward.core import LogBackward
        return LogBackward.apply(self)

    @staticmethod
    def softmax(x: 'Tensor'):
        from nn.backward.activations import SoftmaxBackward
        return SoftmaxBackward.apply(x)

    # main function for engine
    def backward(self):
        from nn.engine import Engine
        engine = Engine()
        engine.backward(self)

    def __str__(self):
        return f"Tensor: {self.data}, requires_grad: {self.requires_grad}"\

    def unsqueeze(self, dim: int):
        from nn.backward.core import UnsqueezeBackward
        return UnsqueezeBackward.apply(self, dim)

    def long(self):
        return Tensor(self.data.astype(np.int64), requires_grad=self.requires_grad)

    def float(self):
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

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
    def __init__(self, data: np.ndarray, requires_grad=True, grad_fn=None):
        super().__init__(data, requires_grad=requires_grad, grad_fn=grad_fn)
