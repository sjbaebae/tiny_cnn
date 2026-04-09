import numpy as np

# import linear layer
from nn.layers.main import Linear, Relu, Gelu, Silu, Sigmoid, Tanh

# import Tensor
from tensor import Tensor, no_grad

from nn.layers import Module

class MLP(Module):
    def __init__(self, in_features, hidden_features, out_features, num_hidden_layers, bias = True, activation="relu"):
        super().__init__()
        self.l1 = Linear(in_features, hidden_features)
        self.l2 = Linear(hidden_features, hidden_features)
        self.l3 = Linear(hidden_features, hidden_features)
        self.l4 = Linear(hidden_features, out_features)
        
        if activation == "relu":
            self.activation = Relu()
        elif activation == "gelu":
            self.activation = Gelu()
        elif activation == "silu":
            self.activation = Silu()
        elif activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()

    def forward(self, x: Tensor):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor")
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.l4(x)
        x = self.activation(x)

        return x

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":
    with no_grad():
        basic_mlp = MLP(in_features=20, out_features=2)
        print(basic_mlp(Tensor(np.random.rand(20))))