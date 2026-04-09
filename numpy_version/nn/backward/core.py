from .main import Node, Edge, Function
import numpy as np

def get_edge(t):
    if getattr(t, "requires_grad", False) is False:
        return None
    if t.grad_fn is not None:
        return Edge(t.grad_fn, 0)
    return Edge(AccumulateGrad(t), 0)

def reduce_to_shape(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class AccumulateGrad(Node):
    def __init__(self, tensor):
        super().__init__(edges=(), saved_tensors=(tensor,))
    
    def backward(self, grad_in: np.ndarray):
        tensor = self.saved_tensors[0]
        if tensor.grad is None:
            tensor.grad = grad_in.copy()
        else:
            tensor.grad = tensor.grad + grad_in

class ReshapeBackward(Function):
    @staticmethod
    def forward(data, shape):
        return data.reshape(shape)

    def __init__(self, tensor, shape):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.data_shape = tensor.shape
    
    def backward(self, grad_in: np.ndarray):
        return grad_in.reshape(self.data_shape),

class ViewBackward(Function):
    @staticmethod
    def forward(data, shape):
        return data.reshape(shape)

    def __init__(self, tensor, shape):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.data_shape = tensor.shape
    
    def backward(self, grad_in: np.ndarray):
        return grad_in.reshape(self.data_shape),

class SliceBackward(Function):
    @staticmethod
    def forward(data, key):
        return data[key]

    def __init__(self, tensor, key):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.slice_indices = key
        self.data_shape = tensor.shape
    
    def backward(self, grad_in: np.ndarray):
        grad = np.zeros(self.data_shape)
        grad[self.slice_indices] = grad_in
        return grad,

class PermuteBackward(Function):
    @staticmethod
    def forward(data, permuted_dims):
        return data.transpose(permuted_dims)

    def __init__(self, tensor, permuted_dims):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.permute_dims = permuted_dims
    
    def backward(self, grad_in: np.ndarray):
        return grad_in.transpose(np.argsort(self.permute_dims)),

class AddBackward(Function):
    @staticmethod
    def forward(left, right):
        return left + right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)))
        self.left_shape = getattr(left, "shape", ())
        self.right_shape = getattr(right, "shape", ())

    def backward(self, grad_in: np.ndarray):
        return reduce_to_shape(grad_in, self.left_shape), reduce_to_shape(grad_in, self.right_shape)

class SubBackward(Function):
    @staticmethod
    def forward(left, right):
        return left - right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)))
        self.left_shape = getattr(left, "shape", ())
        self.right_shape = getattr(right, "shape", ())

    def backward(self, grad_in: np.ndarray):
        return reduce_to_shape(grad_in, self.left_shape), reduce_to_shape(-grad_in, self.right_shape)

class MulBackward(Function):
    @staticmethod
    def forward(left, right):
        return left * right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)), saved_tensors=(left, right))

    def backward(self, grad_in: np.ndarray):
        left_data = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        right_data = getattr(self.saved_tensors[1], "data", self.saved_tensors[1])
        return reduce_to_shape(right_data * grad_in, getattr(left_data, "shape", ())), reduce_to_shape(left_data * grad_in, getattr(right_data, "shape", ()))

class DivBackward(Function):
    @staticmethod
    def forward(left, right):
        return left / right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)), saved_tensors=(left, right))

    def backward(self, grad_in: np.ndarray):
        left_data = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        right_data = getattr(self.saved_tensors[1], "data", self.saved_tensors[1])
        return reduce_to_shape(1/right_data * grad_in, getattr(left_data, "shape", ())), reduce_to_shape(-left_data / right_data ** 2 * grad_in, getattr(right_data, "shape", ()))

class PowBackward(Function):
    @staticmethod
    def forward(left, right):
        return left ** right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)), saved_tensors=(left, right))

    def backward(self, grad_in: np.ndarray):
        left_data = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        right_data = getattr(self.saved_tensors[1], "data", self.saved_tensors[1])
        return reduce_to_shape(right_data * left_data ** (right_data - 1) * grad_in, getattr(left_data, "shape", ())), reduce_to_shape(left_data ** right_data * np.log(left_data) * grad_in, getattr(right_data, "shape", ()))

class NegBackward(Function):
    @staticmethod
    def forward(left):
        return -left

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),))
        self.data_shape = getattr(left, "shape", ())
    
    def backward(self, grad_in: np.ndarray):
        return reduce_to_shape(-grad_in, self.data_shape), 

class AbsBackward(Function):
    @staticmethod
    def forward(left):
        return np.abs(left)

    def __init__(self, left):
        super().__init__(edges=(get_edge(left),), saved_tensors=(left,))
        self.data_shape = getattr(left, "shape", ())
    
    def backward(self, grad_in: np.ndarray):
        data = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        return reduce_to_shape(np.sign(data) * grad_in, self.data_shape),

class MatmulBackward(Function):
    @staticmethod
    def forward(left, right):
        return left @ right

    def __init__(self, left, right):
        super().__init__(edges=(get_edge(left), get_edge(right)), saved_tensors=(left, right))
    
    def backward(self, grad_in: np.ndarray):
        left_data = getattr(self.saved_tensors[0], "data", self.saved_tensors[0])
        right_data = getattr(self.saved_tensors[1], "data", self.saved_tensors[1])

        grad_left = grad_in @ right_data.T
        grad_right = left_data.T @ grad_in
        return reduce_to_shape(grad_left, getattr(left_data, "shape", ())), reduce_to_shape(grad_right, getattr(right_data, "shape", ()))

class SumBackward(Function):
    @staticmethod
    def forward(data, axis=None, keepdims=False):
        return data.sum(axis=axis, keepdims=keepdims)

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.data_shape = tensor.shape
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad_in: np.ndarray):
        if self.axis is None:
            # Summed over all elements: broadcast scalar gradient to original shape
            return grad_in * np.ones(self.data_shape),
        
        # If summarized over a specific axis and dimensions were squeezed out,
        # we reshape grad_in so it can be cleanly broadcasted against np.ones
        if not self.keepdims:
            grad_in = np.expand_dims(grad_in, axis=self.axis)
            
        return grad_in * np.ones(self.data_shape),

class UnsqueezeBackward(Function):
    @staticmethod
    def forward(data, dim):
        return np.expand_dims(data, axis=dim)

    def __init__(self, tensor, dim):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.data_shape = tensor.shape
        self.dim = dim

    def backward(self, grad_in: np.ndarray):
        return np.squeeze(grad_in, axis=self.dim), 

class LogBackward(Function):
    @staticmethod
    def forward(data, base=np.e):
        # np.log(e) = 1, so np.log(data) / np.log(e) = np.log(data) which is the normal log base e form.
        # if base is not e, it is equivalent to np.log(data) / np.log(base)
        return np.log(data) / np.log(base)
    
    def __init__(self, tensor, base=np.e):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=(tensor,))
        self.base = base
        
    def backward(self, grad_in: np.ndarray):
        # gradient of ln is 1/x. For any of other bases, it is 1/(x * ln(base))
        grad = 1 / (self.saved_tensors[0].data * np.log(self.base))
        return grad * grad_in,
        
        
        