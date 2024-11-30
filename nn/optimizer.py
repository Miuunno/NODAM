class Optimizer:
    def __call__(self):
        return None

class SGD(Optimizer):
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd

    def __call__(self, layer, gradient):
        gradient = self.lr * (gradient.T + self.wd * layer.coeff)
        return gradient

class Identity(Optimizer):
    def __call__(self, layer, gradient):
        return gradient