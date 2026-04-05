from nn import Module
import numpy as np

class Activation(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray):
        pass

#############################
#          CLASSES          #
#############################

class Relu(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray):
        return relu(x)

    