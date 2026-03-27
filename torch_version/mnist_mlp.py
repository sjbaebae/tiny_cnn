import torch
from torch import nn
import torch.nn.functional as F

class Linear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: str = "relu") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid 
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"Activation function {activation} not supported")
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, num_hidden_layers: int, activation: str = "relu") -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(in_features, hidden_features, activation=activation))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Linear(hidden_features, hidden_features, activation=activation))
        self.layers.append(Linear(hidden_features, out_features, activation="none"))

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x