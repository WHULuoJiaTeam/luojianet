import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
import luojianet_ms.context as context
from luojianet_ms import Tensor
import luojianet_ms.ops as ops
import numpy as np
import pytest

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.tensor_add_v2 = ops.TensorAddV2()

    def forward(self, x1, x2):
        return self.tensor_add_v2(x1, x2)

class Grad(nn.Module):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=True)
        self.network = network

    def forward(self, x1, x2, sens):
        gout = self.grad(self.network)(x1, x2, sens)
        return gout

def test_grad_net():
    x1 = Tensor(np.ones((3, 4), np.float32))
    x2 = Tensor(np.ones((3, 4), np.float32))
    sens = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    grad = Grad(Net())
    dx = grad(x1, x2, sens)
    print("\n dx[0]: ", dx[0].asnumpy())
