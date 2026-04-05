import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        if self.is_leaf:
            self.grad = np.zeros_like(data)
        self.grad_fn = None
        self._accumulate_grad = AccumulateGrad(self) if requires_grad and grad_fn is None else None

    def backward(self, grad_in: np.ndarray = np.array([1])):
        self.grad_fn(grad_in)
    
    def __add__(self, other):
        raw_data = self.data + other.data
        result = Tensor(raw_data, requires_grad = self.requires_grad or other.requires_grad, grad_fn = AddBackward(self, other))
        return result
    
    def __sub__(self, other):
        raw_data = self.data - other.data
        result = Tensor(raw_data, requires_grad = self.requires_grad or other.requires_grad, grad_fn = SubBackward(self, other))
        return result
    
    def __mul__(self, other):
        result = Tensor(self.data * other.data)
        result.grad_fn = MulBackward(self, other)
        return result
    
    def __truediv__(self, other):
        result = Tensor(self.data / other.data)
        result.grad_fn = DivBackward(self, other)
        return result
    
    def __pow__(self, other):
        result = Tensor(self.data ** other.data)
        result.grad_fn = PowBackward(self, other)
        return result
    
    def __neg__(self):
        return Tensor(-self.data)
    
    def __abs__(self):
        return Tensor(np.abs(self.data))
    
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
        return Tensor(self.data @ other.data)
    
    def reshape(self, shape: tuple): 
        self.data.reshape(tuple)

    @property
    def train(self):
        self.training = True
    
    @property
    def eval(self):
        self.training = False

    @property
    def is_leaf(self):
        return self.requires_grad and self.grad_fn is None

class Module:
    def __init__(self):
        pass
    def forward(self, x: np.ndarray):
        pass
    def __call__(self, x):
        return self.forward(x)
    def train(self):
        self.training = True
    def eval(self):
        self.training = False

# Parameter (autotrack weights)
class Parameter:
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.requires_grad = requires_grad
        
