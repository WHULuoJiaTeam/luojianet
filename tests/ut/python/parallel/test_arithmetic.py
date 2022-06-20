# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import luojianet_ms as ms
import luojianet_ms.nn as nn
from luojianet_ms import Parameter, Tensor, context
from luojianet_ms.common.api import _cell_graph_executor
from luojianet_ms.ops import composite as C
from luojianet_ms.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Module):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def call(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Module):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def call(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b):
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


def test_matmul_sub():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_matmul_mod():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mod net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mod = P.Mod().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mod(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_matmul_floormod():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floormod net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floormod = P.FloorMod().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floormod(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_atan2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-atan2 net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.atan2 = P.Atan2().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.atan2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_divNoNan():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-divNoNan net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.divNoNan = P.DivNoNan().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.divNoNan(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_logicaland():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-logical_and net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.equal = P.Equal().shard(strategy2)
            self.notequal = P.NotEqual().shard(strategy2)
            self.logical = P.LogicalAnd().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out1 = self.equal(out, b)
            out = self.matmul(x, y)
            out2 = self.notequal(out, b)
            out = self.logical(out1, out2)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_logicalor():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-logical_or net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.equal = P.Equal().shard(strategy2)
            self.notequal = P.NotEqual().shard(strategy2)
            self.logical = P.LogicalOr().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out1 = self.equal(out, b)
            out = self.matmul(x, y)
            out2 = self.notequal(out, b)
            out = self.logical(out1, out2)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_sub_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_sub_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_greater_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-greater broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.greater = P.Greater().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_greater_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-greater broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.greater = P.Greater().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv_broadcast():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv_broadcast2():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def call(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_assign_sub():
    """
    Feature: distribute operator sub in auto parallel.
    Description: mul-assign_sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.AssignSub()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def call(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Module):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def call(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Module):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def call(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign_add():
    """
    Feature: distribute operator sub in auto parallel.
    Description: mul-assign_add net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.AssignAdd()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def call(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Module):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def call(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Module):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def call(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign():
    """
    Feature: distribute operator sub in auto parallel.
    Description: mul-assign_sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.Assign()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def call(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Module):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def call(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Module):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def call(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)
