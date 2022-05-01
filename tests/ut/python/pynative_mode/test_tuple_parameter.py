import numpy as np

import luojianet_ms.nn as nn
from luojianet_ms import context, Tensor
from luojianet_ms.ops import operations as P
from luojianet_ms.ops import composite as C



def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class Block1(nn.Module):
    """ Define Module with tuple input as parameter."""

    def __init__(self):
        super(Block1, self).__init__()
        self.mul = P.Mul()

    def call(self, tuple_xy):
        x, y = tuple_xy
        z = self.mul(x, y)
        return z

class Block2(nn.Module):
    """ definition with tuple in tuple output in Module."""

    def __init__(self):
        super(Block2, self).__init__()
        self.mul = P.Mul()
        self.add = P.Add()

    def call(self, x, y):
        z1 = self.mul(x, y)
        z2 = self.add(z1, x)
        z3 = self.add(z1, y)
        return (z1, (z2, z3))

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.block = Block1()

    def call(self, x, y):
        res = self.block((x, y))
        return res


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.add = P.Add()
        self.block = Block2()

    def call(self, x, y):
        z1, (z2, z3) = self.block(x, y)
        res = self.add(z1, z2)
        res = self.add(res, z3)
        return res

def test_net():
    x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32) * 2)
    y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32) * 3)
    net1 = Net1()
    grad_op = C.GradOperation(get_all=True)
    output = grad_op(net1)(x, y)
    assert np.all(output[0].asnumpy() == y.asnumpy())
    assert np.all(output[1].asnumpy() == x.asnumpy())

    net2 = Net2()
    output = grad_op(net2)(x, y)
    expect_x = np.ones([1, 1, 3, 3]).astype(np.float32) * 10
    expect_y = np.ones([1, 1, 3, 3]).astype(np.float32) * 7
    assert np.all(output[0].asnumpy() == expect_x)
    assert np.all(output[1].asnumpy() == expect_y)
