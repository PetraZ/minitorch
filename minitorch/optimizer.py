from .network import NetWork


class Optimizer:
    def step(self, net: NetWork) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def step(self, net: NetWork) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
