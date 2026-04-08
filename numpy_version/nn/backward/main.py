import numpy as np
# backward op (node) -> edge is then tensor. Backward ops lead back to tensors -> to ops. Directed acyclic graph

class Edge:
    def __init__(self, fn, input_nr: int):
        self.fn = fn
        self.input_nr = input_nr

class Node:
    def __init__(self, next_edges: tuple[Edge, ...], saved_tensors=()):
        self.next_edges = next_edges
        self.saved_tensors = saved_tensors

    def apply(self, grad_in):
        # apply to our set of edges iteratively
        raise NotImplementedError