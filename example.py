import numpy as np

from minitorch.layer import Linear, Tanh
from minitorch.network import NetWork
from minitorch.train import train

inputs = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

targets = np.array(
    [
        [0],
        [1],
        [1],
        [0],
    ]
)


net = NetWork([Linear(inp_size=2, out_size=3), Tanh(), Linear(inp_size=3, out_size=1)])
train(net, inputs, targets)
for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
