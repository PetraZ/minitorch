from .data import DataIterator
from .loss import MSE, Loss
from .network import NetWork
from .optimizer import SGD, Optimizer
from .tensor import Tensor


def train(
    net: NetWork,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 500,
    iterator: DataIterator = DataIterator(),
    loss: Loss = MSE(),
    optim: Optimizer = SGD(),
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in iterator(inputs, targets):
            preds = net.forward(batch.inputs)
            epoch_loss += loss.loss(preds, batch.targets)
            grad = loss.grad(preds, batch.targets)
            net.backward(grad)
            optim.step(net)
        print(epoch, epoch_loss)
