from .main import Node, Edge
from ...tensor import Tensor
import numpy as np

# helper functions. 

# get_edge to determine when to stop / accumulate, etc in our graph
def get_edge(t: Tensor):
    if not t.requires_grad:
        return None
    if t.grad_fn is not None:
        return Edge(t.grad_fn, 0)
    return Edge(AccumulateGrad(t), 0)

# Reduce to the proper shape to avoid shape issues / braodcasting

def reduce_to_shape(grad, shape):
    # sum extra leading dims
    while len(grad.shape) > len(shape):
        # remove leading dims. Align on right end
        grad = grad.sum(axis=0)

    # sum broadcasted dims
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# Base Accumulate
class AccumulateGrad(Node):
    def __init__(self, tensor: Tensor):
        super().__init__(edges=(), saved_tensors=(tensor,))
        # this is a leaf node, so it has no next edges
    
    def apply(self, grad_in: np.ndarray):
        tensor = self.saved_tensors[0]
        # grads will not always exist
        if tensor.grad is None:
            tensor.grad = grad_in.copy()
        else:
            tensor.grad = tensor.grad + grad_in

# Reshape Helpers
class ReshapeBackward(Node):
    def __init__(self, tensor: Tensor):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.original_shape = tensor.data.shape

    
    def apply(self, grad_in: np.ndarray):
        return grad_in.reshape(self.original_shape)

class ViewBackward(Node):
    def __init__(self, tensor: Tensor):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.original_shape = tensor.data.shape
    
    def apply(self, grad_in: np.ndarray):
        # view does not change the data, so we just return the same gradient
        return grad_in.reshape(self.original_shape)

class SliceBackward(Node):
    def __init__(self, tensor: Tensor):
        super().__init__(edges=(get_edge(tensor),), saved_tensors=())
        self.slice_indices = tensor.slice_indices
    
    def apply(self, grad_in: np.ndarray):
        # create zero array with original shape
        grad = np.zeros_like(self.original_shape)
        # slice it. Since the rest are just 0 gradients, only slice regions will receive the target gradient
        grad[self.slice_indices] = grad_in
        return grad
        
        

class PermuteBackward(Node):
    def __init__(self, tensor: Tensor, permuted_dims):
        super().__init__(edges=(Edge(tensor.grad_fn, 0),), saved_tensors=())
        self.permute_dims = permuted_dims
    
    def apply(self, grad_in: np.ndarray):
        return grad_in.transpose(self.permute_dims)
        

# note these are binary ops, so only ever 2 inputs
class AddBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges=(Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)))

    def apply(self, grad_in: np.ndarray):
        # gradients 0 and 1
        return grad_in, grad_in

class SubBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)))

    def apply(self, grad_in: np.ndarray):
        return grad_in, -grad_in
            

# HADAMAARD PRODUCT
class MulBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)),
        saved_tensors = (left, right))
    def apply(self, grad_in: np.ndarray):
        return self.saved_tensors[1].data * grad_in, self.saved_tensors[0].data * grad_in

class DivBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)),
        saved_tensors = (left, right))

    def apply(self, grad_in: np.ndarray):
        return 1/self.saved_tensors[1].data * grad_in, -self.saved_tensors[0].data / self.saved_tensors[1].data ** 2 * grad_in

class PowBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)),
        saved_tensors = (left, right))
        self.left_data = left.data
        self.right_data = right.data
    def apply(self, grad_in: np.ndarray):
        return self.right_data * self.left_data ** (self.right_data - 1) * grad_in, self.left_data ** self.right_data * np.log(self.left_data) * grad_in

class NegBackward(Node):
    def __init__(self, left: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0),))
    
    def apply(self, grad_in: np.ndarray):
        return -grad_in

class AbsBackward(Node):
    def __init__(self, left: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0),), saved_tensors = (left,))
    
    def apply(self, grad_in: np.ndarray):
        return np.sign(self.saved_tensors[0].data) * grad_in

class MatmulBackward(Node):
    def __init__(self, left: Tensor, right: Tensor):
        super().__init__(edges = (Edge(left.grad_fn, 0), Edge(right.grad_fn, 1)),
        saved_tensors = (left, right))

        self.left_data = left.data
        self.right_data = right.data
    
    def apply(self, grad_in: np.ndarray):
        # grad_in: (batch, out_features)
        # left: (batch, in_features)
        # right: (in_features, out_features)
        # grad_left: (batch, in_features)
        # grad_right: (in_features, out_features)

        grad_left = grad_in @ self.right_data.T
        grad_right = self.left_data.T @ grad_in
        return grad_left, grad_right
        
    
    
    