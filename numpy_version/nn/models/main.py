from ..layers import Module, Linear

class MLP(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.l1 = Linear(in_features, out_features)
        self.l2 = Linear(out_features, out_features)
        self.l3 = Linear(out_features, out_features)
        