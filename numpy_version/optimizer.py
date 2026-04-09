
import numpy as np
from tensor import Parameter

class Optim:
    def __init__(self, params: list[Parameter]):
        self.params = params
        pass
    def step(self):
        # specific to each optimizer
        pass
    def zero_grad(self):
        # will go through the list of parameters and zerograd them all
        for param in self.params:
            if param.requires_grad:
                # zero gradient array
                param.grad = np.zeros(param.shape)

class SGD(Optim):
    def __init__(self, params: list[Parameter], lr: float = 0.01):
        super().__init__(params)
        # SGD formula per step: delta = - learning_rate * gradient
        self.lr = lr
    
    def step(self):
        for param in self.params:
            if param.requires_grad:
                param.data = param.data - self.lr * param.grad

class Momentum(Optim):
    def __init__(self, params: list[Parameter], lr: float = 0.01, momentum = 0.9):
        super().__init__(params)
        # momentum formula per step: delta = momentum * v_{t-1} + learning_rate * gradient
        # this is analgous to an EMA of the gradient. 

        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in params]
    
    def step(self):
        for i, (param, v) in enumerate(zip(self.params, self.v)):
            if param.requires_grad:
                v = self.momentum * v + self.lr * param.grad
                param.data = param.data - v
                self.v[i] = v

class Adam(Optim):
    def __init__(self, params: list[Parameter], lr: float = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0.01):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros(param.shape) for param in params]
        self.v = [np.zeros(param.shape) for param in params]
        self.t = 0
        self.weight_decay = weight_decay
    
    def step(self):
        self.t += 1
        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            if param.requires_grad:
                grad = param.grad + self.weight_decay * param.data # weight decay
                m = self.beta1 * m + (1 - self.beta1) * grad # first moment
                v = self.beta2 * v + (1 - self.beta2) * grad ** 2 # second moment of gradient
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.m[i] = m
                self.v[i] = v

class AdamW(Optim):
    def __init__(self, params: list[Parameter], lr: float = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0.01):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = [np.zeros(param.shape) for param in params]
        self.v = [np.zeros(param.shape) for param in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            if param.requires_grad:
                m = self.beta1 * m + (1 - self.beta1) * param.grad # first moment
                v = self.beta2 * v + (1 - self.beta2) * param.grad ** 2 # second moment of gradient
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                # in AdamW we directly add weight decay to the gradient rather than doing the adaptive scaling of the gradient
                param.data = param.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param.data)
                self.m[i] = m
                self.v[i] = v
                
        