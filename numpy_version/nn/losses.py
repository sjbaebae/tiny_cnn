import numpy as np
from tensor import Tensor

class Loss:
    def __init__(self):
        pass
    
    def __call__(self, y_pred, y_true):
        # should return a tensor
        pass

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y_pred: Tensor, y_true: Tensor):
        # y_pred is logits
        # y_true is one hot
        # cross entropy loss

        # cross entropy is -1 * sum(y_true * log(softmax(y_pred)))

        summed_logits: Tensor = (y_true * (Tensor.softmax(y_pred)).log()).sum()

        return -summed_logits
        
        