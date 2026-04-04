from nn import Module
import numpy as np

class MSE(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return np.mean((y_pred - y_true)**2)
    