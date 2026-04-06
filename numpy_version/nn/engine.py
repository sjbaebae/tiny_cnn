from .backward import Node, Edge
from ..tensor import Tensor
from collections import deque, defaultdict
import numpy as np
# backward gradients engine

class Engine:
    # engine is initialized with the loss.backward() call. Which is a tensor with grad_fn
    def __init__(self):
        pass

    @staticmethod
    def get_dependencies(root_tensor: Tensor):
        visited = set()
        dependencies = defaultdict(int) # the # of dependencies of each node. Note that since we know the local connections. Combined with the global count, it is enough to know when to run the backward pass.

        def dfs(node: Node):
            if node is None or id(node) in visited:
                return
            visited.add(id(node))
            for edge in node.next_edges:
                # edge.fn points to the next Node backward func
                if edge.fn is not None:
                    dependencies[edge.fn] += 1
                    dfs(edge.fn)
        
        if root_tensor.grad_fn is not None:
            # this is the starting node
            dfs(root_tensor.grad_fn)
        
        return dependencies

    def backward(self, root: Tensor, initial_grad: np.ndarray = np.array([1])):
        dependencies = self.get_dependencies(root)
        queue = deque()
        not_ready = defaultdict(list)

        queue.append((root.grad_fn, initial_grad))

        while queue:
            node, inbuf = queue.pop()

            if not isinstance(node, Node):
                # no grads to push
                continue

            outputs = node.apply(inbuf)

            # guard against tuple breaking
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            for idx, edge in enumerate(node.next_edges):
                if edge is None:
                    continue
                edge: Edge
                parent: Node = edge.fn
                not_ready[parent].append((idx, outputs[idx]))
                dependencies[parent] -= 1
                if dependencies[parent] == 0:
                    grad_tuples = not_ready[parent]
                    grad_arrays = [g for _, g in sorted(grad_tuples, key=lambda x: x[0])]
                    # only when all dependencies cleared we push gradient to be used on a node. 
                    total = grad_arrays[0]
                    # numpy will auto handle shape braodcasting per summation.
                    for g in grad_arrays[1:]:
                        total = total + g
                    del not_ready[parent]
                    queue.append((parent, total))
                


    
    