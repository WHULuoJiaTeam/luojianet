import numpy as np
import luojianet_ms.nn as nn
import luojianet_ms.context as context
from luojianet_ms import Tensor
import luojianet_ms.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transposenew = ops.operations.TransposeNew()

    def forward(self, data):
        return self.transposenew(data, (1, 0))

def test_net():
    x = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    transposenew = Net()
    output = transposenew(Tensor(x))
    print("output: ", output)


test_net()
