import numpy as np
import models

def relu(x): 
    x[x < 0] = 0
    return x

class MLP(models.NN):
    def __init__(self, in_features, out_features, bias = True, activation="relu"):
        super().__init__()
        self.weight = np.random.rand(in_features, out_features)

        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = None

    def forward(self, x: np.ndarray):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias

        return relu(x)

    def __call__(self, x):
        return self.forward(x)

basic_mlp = MLP(in_features=20, out_features=2)
print(basic_mlp(np.random.rand(20)))