import numpy as np
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor
from luojianet_ms import ParameterTuple, Parameter
from luojianet_ms import dtype as mstype


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        # self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z') #z为输入变量
        #如果不需要对z求导，可以将require_grad=False
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z', requires_grad=True)

    def call(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Module):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=False) #get_all=True for gradient w.r.t x, y

    def call(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

#gradient with respect to z
x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)

class GradNetWrtW(nn.Module):
    def __init__(self, net):
        super(GradNetWrtW, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def call(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

output = GradNetWrtW(Net())(x, y)
print(output)

import numpy as np
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor
from luojianet_ms import ParameterTuple, Parameter
from luojianet_ms import dtype as mstype
from luojianet_ms.ops import stop_gradient

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.matmul = ops.MatMul()

    def call(self, x, y):
        out1 = self.matmul(x, y)
        out2 = self.matmul(x, y)
        out2 = stop_gradient(out2) #stop gradient so that out2 has no effect on updating the gradient
        out = out1 + out2
        return out

class GradMyNetWrtX(nn.Module):
    def __init__(self, net):
        super(GradMyNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def call(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradMyNetWrtX(MyNet())(x, y)
print("stop gradient result: ")
print(output)