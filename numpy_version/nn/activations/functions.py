import numpy as np

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
    # subtract max for numerical stability
    x_shift = x - np.max(x, axis = 1, keepdims = True)
    return np.exp(x_shift) / np.sum(np.exp(x_shift), axis = 1, keepdims = True)