import numpy as np
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P

context.set_context(enable_graph_kernel=False, save_graphs=False, mode=context.PYNATIVE_MODE, device_target="Ascend")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.host = P.TensorShape()
        self.t1 = Tensor(np.random.randn(16).astype(np.int32))

    def forward(self):
        return self.host(self.t1)

def test_net():
    """
    Feature: test host kernel in pynative mode
    Description: get shape
    Expectation: success
    """
    net = Net()
    out1 = net()
    print(out1.asnumpy())
