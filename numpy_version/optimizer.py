from nn import Module, Parameter

class Optim:
    def __init__(self, params: list[Parameter]):
        self.params = params
        pass
    def step(self):
        pass
    def zerograd(self):
        pass