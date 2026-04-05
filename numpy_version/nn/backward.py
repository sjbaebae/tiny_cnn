from .main import Node
import numpy as np

# note these are binary ops, so only ever 2 inputs
class AddBackward(Node):
    def __init__(self, left, right):
        super().__init__(left, right)

    def apply(self, grad_in: np.ndarray):
        return grad_in, grad_in

class SubBackward(Node):
    def __init__(self, left, right):
        super().__init__(left, right)

    def apply(self, grad_in: np.ndarray):
        return grad_in, -grad_in
            

# HADAMAARD PRODUCT
class MulBackward(Node):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left_data = left.data
        self.right_data = right.data
    def apply(self, grad_in: np.ndarray):
        return self.right_data * grad_in, self.left_data * grad_in

class DivBackward(Node):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def apply(self, grad_in: np.ndarray):
        return 1/self.right * grad_in, -self.left / self.right ** 2 * grad_in

class PowBackward(Node):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left_data = left.data
        self.right_data = right.data
    def apply(self, grad_in: np.ndarray):
        return self.right_data * self.left_data ** (self.right_data - 1) * grad_in, self.left_data ** self.right_data * np.log(self.left_data) * grad_in

