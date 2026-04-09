import numpy as np

# global autograd tracking guard
_grad_enabled = True

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self._prev
# backward op (node) -> edge is then tensor. Backward ops lead back to tensors -> to ops. Directed acyclic graph

class Edge:
    def __init__(self, fn, input_nr: int):
        self.fn = fn
        self.input_nr = input_nr

class Node:
    def __init__(self, edges: tuple[Edge, ...] = (), saved_tensors=()):
        self.next_edges = edges
        self.saved_tensors = saved_tensors

    def backward(self, grad_in):
        raise NotImplementedError

class Function(Node):
    @classmethod
    def apply(cls, *args, **kwargs):
        from tensor import Tensor
        global _grad_enabled
        
        # Check if gradient tracking is required
        requires_grad = _grad_enabled and any(isinstance(arg, Tensor) and arg.requires_grad for arg in args)
        
        # Unwrap data from Tensors
        data_args = [arg.data if isinstance(arg, Tensor) else arg for arg in args]
        
        # Compute forward pass
        raw_result = cls.forward(*data_args, **kwargs)
        
        # Wrap into Tensor
        out = Tensor(raw_result, requires_grad=requires_grad)
        
        # Build the backward node
        if requires_grad:
            # should convert any tensor that is not a tensor to a tensor. This is to ensure backward gradients are computed correctly.
            args = [Tensor(arg) if not isinstance(arg, Tensor) else arg for arg in args]
            out.grad_fn = cls(*args, **kwargs)
            
        return out