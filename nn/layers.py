import jax.numpy as jp
from .utils import normal, shuffle, choice
from tqdm import tqdm

# Non-Trivial Layers
class Module:
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.criterion = loss
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.criterion = loss
    def fit(self, x, y, batch_size, epochs, verbose=0):
        for epoch in range(epochs):
            print(f"----> Start epoch: {epoch}")
            extra = len(x) % batch_size
            idx = choice(shuffle(len(x)), (len(x) - extra)).reshape(-1, batch_size)
            for id in idx:
                out = self.__call__(x[id])
                loss, grad = self.criterion(out, y[id])
                self.backward(grad)
                self.optimize()
                self.zero_grad()


class Layer:
    def __call__(self, x):
        return None

class Model(Module):
    def __init__(self):
        super().__init__(None, None)
    def backward(self, gradient):
        for layer in reversed(self.__getstate__()):
            layer = self.__getattribute__(layer)
            if isinstance(layer, Layer):
                if hasattr(layer, 'local_gradient'):
                    layer.local_gradient(gradient)
                    gradient = layer.backward_gradient(gradient)
    def optimize(self):
        for layer in reversed(self.__getstate__()):
            layer = self.__getattribute__(layer)
            if isinstance(layer, Layer) and hasattr(layer, 'coeff'):
                for grad in layer.grad:
                    layer.coeff = layer.coeff - self.optimizer(layer, grad.reshape(-1, 1))
    def zero_grad(self):
        for layer in reversed(self.__getstate__()):
            layer = self.__getattribute__(layer)
            if isinstance(layer, Layer) and hasattr(layer, 'coeff'):
                layer.grad = None

# Main Function Layers
class Linear(Layer):
    def __init__(self, input, output):
        super().__init__()
        self.coeff = normal(input + 1, output)
        self.x = None
        self.grad = None
    def __call__(self, x):
        shape = [i for i in x.shape]
        shape[-1] = 1
        res = jp.ones(shape)
        x = jp.concatenate((x, res), axis=-1)
        self.x = x
        return jp.dot(x, self.coeff)
    def local_gradient(self, dx_dphi):
        self.grad = jp.array([dx * self.x[..., idx] for idx, dx in enumerate(dx_dphi)])
    def backward_gradient(self, dx_dphi):
        return jp.array([dx * self.coeff[idx].sum() for idx, dx in enumerate(dx_dphi)])

# Activation Layer
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None
    def __call__(self, x):
        self.x = x
        return (x + jp.abs(x)) / 2
    def backward_gradient(self, dx_dphi):
        return jp.array([dx * self.gradient(self.x[..., idx]).sum(axis=-1) for idx, dx in enumerate(dx_dphi)])
    def gradient(self, x):
        return jp.where(x > 0, 1, x)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.x = None
    def __call__(self, x):
        self.x = x
        return (1+jp.exp(-x)) ** (-1)
    def backward_gradient(self, dx_dphi):
        return jp.array([dx * self.gradient(self.x[..., idx]).sum(axis=-1) for idx, dx in enumerate(dx_dphi)])
    def gradient(self, x):
        return (2 * jp.exp(2*x) + jp.exp(x)) * ((jp.exp(x) + 1) ** (-2))

class GELU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None
    def __call__(self, x):
        self.x = x
        return 0.5*x*(1 + jp.tanh((2/jp.pi) ** (1/2) * (x + 0.044715 * (x ** 3))))
    def backward_gradient(self, dx_dphi):
        return jp.array([dx * self.gradient(self.x[..., idx]).sum(axis=-1) for idx, dx in enumerate(dx_dphi)])
    def gradient(self, x):
        one = (2 / jp.pi) ** (1/2) * (x + 0.044715 * (x**3))
        return 0.5 * (1 + jp.tanh(one) + x * (1 - jp.tanh(one) **2)*(1 + 0.044715*3*(x**2))*((2/jp.pi) ** (1/2)))

# Normalize Layers
class LayerNorm(Layer):
    def __init__(self, shape, eps):
        self.coeff = normal(shape + 1, shape)
        self.eps = eps
        self.x = None
        self.grad = None
    def __call__(self, x):
        self.x = x
        shape = [i for i in x.shape]
        shape[-1] = 1
        res = jp.ones(shape)
        x = jp.concatenate((x, res), axis=1)
        self.x = ((x - x.mean(axis=-1))/ (x.var(axis=-1) + self.eps))
        return self.x @ self.coeff
    def local_gradient(self, dx_dphi):
        self.grad = jp.array([dx * self.x[..., idx] for idx, dx in enumerate(dx_dphi)])
    def backward_gradient(self, dx_dphi):
        return jp.array([dx * self.coeff[idx].sum() for idx, dx in enumerate(dx_dphi)])






