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
# ============================================================================
""" test_parser_tensor_assign """
import pytest
import numpy as np
import luojianet_ms as ms
from luojianet_ms import context
from luojianet_ms.nn import ReLU
from luojianet_ms.nn import Module
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.ops import operations as P

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_tensor_assign_slice():
    class Net(Module):
        def __init__(self, U):
            super(Net, self).__init__()
            self.relu = ReLU()
            self.U = U

        def call(self, x):
            x = self.relu(x)
            x[..., :2] = U
            return x

    input_np_x = np.random.rand(4, 4, 4)
    input_me_x = Tensor(input_np_x, ms.float32)
    U = 1.0

    net = Net(U)
    out_me = net(input_me_x)
    input_np_x[..., :2] = U

    assert np.allclose(out_me.asnumpy(), input_np_x, rtol=0.01, atol=0.01)

def test_parser_tensor_assign_slice_002():
    class Net(Module):
        def __init__(self, U):
            super(Net, self).__init__()
            self.relu = ReLU()
            self.U = U

        def call(self, x):
            x = self.relu(x)
            x[::, :, :1] = self.U
            return x

    input_np_x = np.random.rand(4, 4, 4)
    input_me_x = Tensor(input_np_x, ms.float32)
    U = 1.0

    net = Net(U)
    out_me = net(input_me_x)
    input_np_x[::, :, :1] = U

    assert np.allclose(out_me.asnumpy(), input_np_x, rtol=0.01, atol=0.01)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_tensor_assign_bool():
    class Net(Module):
        def __init__(self, U):
            super(Net, self).__init__()
            self.relu = ReLU()
            self.U = U

        def call(self, x, tensorB):
            x = self.relu(x)
            x[tensorB] = self.U
            return x

    input_np_x = np.random.rand(4, 4, 4)
    input_me_x = Tensor(input_np_x, ms.float32)
    numpy_B = np.random.randn(4, 4, 4) > 0
    tensor_B = Tensor(numpy_B)
    U = np.array([1])
    net = Net(Tensor(U))

    out_me = net(input_me_x, tensor_B)
    input_np_x[numpy_B] = U

    assert np.allclose(out_me.asnumpy(), input_np_x, rtol=0.01, atol=0.01)

def test_parser_tensor_assign_bool_002():
    class Net(Module):
        def __init__(self, U):
            super(Net, self).__init__()
            self.relu = ReLU()
            self.U = U
            self.fill = P.Fill()

        def call(self, x, tensorB):
            x = self.relu(x)
            x[tensorB] = self.U
            return x

    input_np_x = np.random.rand(2, 2, 2)
    input_me_x = Tensor(input_np_x, ms.float32)
    numpy_B = np.random.randn(2, 2, 2) > 0
    tensor_B = Tensor(numpy_B)
    U = 1

    net = Net(U)
    out_me = net(input_me_x, tensor_B)
    input_np_x[numpy_B] = U
    assert np.allclose(out_me.asnumpy(), input_np_x, rtol=0.01, atol=0.01)
