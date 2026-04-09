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
        # y_pred is logits (batch_size, num_classes)
        # y_true is class indices (batch_size,)
        
        # Determine batch_size and num_classes
        batch_size = y_true.shape[0]
        num_classes = y_pred.shape[1]

        # Produce one-hot using numpy
        one_hot = np.zeros((batch_size, num_classes))
        one_hot[np.arange(batch_size), y_true.data.astype(int)] = 1
        y_true_one_hot = Tensor(one_hot)

        # cross entropy is -1/N * sum(y_true * log(softmax(y_pred)))
        summed_logits: Tensor = (y_true_one_hot * (Tensor.softmax(y_pred) + 1e-9).log()).sum()

        # Expected value, average by batch size
        return -summed_logits / batch_size
        
        