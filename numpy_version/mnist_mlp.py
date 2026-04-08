import numpy as np

# import linear layer
from nn.layers import Linear
from nn.activations.functions import relu, gelu, silu, sigmoid, tanh

# import Tensor
from tensor import Tensor, no_grad

class MLP(models.NN):
    def __init__(self, in_features, out_features, bias = True, activation="relu"):
        super().__init__()
        self.l1 = Linear(in_features, out_features)
        self.l2 = Linear(out_features, out_features)
        self.l3 = Linear(out_features, out_features)
        self.l4 = Linear(out_features, out_features)
        
        if activation == "relu":
            self.activation = relu
        elif activation == "gelu":
            self.activation = gelu
        elif activation == "silu":
            self.activation = silu
        elif activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "tanh":
            self.activation = tanh

    def forward(self, x: Tensor):
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

with no_grad():
    basic_mlp = MLP(in_features=20, out_features=2)
    print(basic_mlp(np.random.rand(20)))