class Loss:
    def __call__(self):
        return None

class MSE(Loss):
    def __call__(self, x, y):
        return ((y - x).T @ (y - x)).mean(), self.gradient(x, y)
    def gradient(self, x, y):
        return 2*x - 2*y

