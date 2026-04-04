from nn import Module
import numpy as np

class Activation(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray):
        pass
    
    def backward(self, x: np.ndarray):
        pass

def relu(x: np.ndarray): 
    return x.clip(min=0)

def gelu(x: np.ndarray): 
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x: np.ndarray): 
    return x * (1 / (1 + np.exp(-x)))

def sigmoid(x: np.ndarray): 
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray): 
    return np.tanh(x)

def softmax(x: np.ndarray): 
    # shape of x: (batch_size, num_classes)
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)