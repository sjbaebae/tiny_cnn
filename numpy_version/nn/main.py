import numpy as np
# backward op (node) -> edge is then tensor. Backward ops lead back to tensors -> to ops. Directed acyclic graph

class Node:
    def __init__(self, edges):
        self.next_edges = tuple(edges)

    def apply(self, grad_in: np.ndarray):
        raise NotImplementedError
        
