# How to build an autograd?

# We want some way to be able to calculate the gradient on the backward pass.
# FOr some Wx + b. Your gradient should be for partial W / partial loss = x. 

# RELU we use identity function
# for partial loss / partial b -> should be 1.

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
    
    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __sub__(self, other):
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data)
    
    def __truediv__(self, other):
        return Tensor(self.data / other.data)
    
    def __pow__(self, other):
        return Tensor(self.data ** other.data)
    
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
    
    def __setattr__(self, name, value):
        self.data[name] = value
    
    def __getattr__(self, name):
        return self.data[name]

    def __matmul__(self, other):
        return Tensor(self.data @ other.data)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

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
        
