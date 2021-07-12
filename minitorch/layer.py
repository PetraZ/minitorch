from typing import Dict

import numpy as np

from .tensor import Tensor


class Layer(object):
    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inp: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, inp_size: int, out_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(inp_size, out_size)
        self.params["b"] = np.random.randn(out_size)

    def forward(self, inp: Tensor) -> Tensor:
        # (n * m(featues)) * (m, o)
        self.inp = inp
        return inp @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        # calculate individual weights here for this layer
        self.grads["b"] = grad.sum(axis=0)
        self.grads["w"] = self.inp.T @ grad
        # backprop the grads here
        return grad @ self.params["w"].T


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp: Tensor) -> Tensor:
        self.inp = inp
        return np.tanh(inp)

    def backward(self, grad: Tensor) -> Tensor:
        return (1 - np.tanh(self.inp) ** 2) * grad
