# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
import pytest
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common import ParameterTuple
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_1_1():
    """
    Feature: Check the names of parameters and the names of inputs of construct.
    Description: If the name of the input of construct is same as the parameters, add suffix to the name of the input.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = Parameter(Tensor([2], ms.float32), name="name_b")

        def construct(self, name_a):
            return self.param_a + self.param_b - name_a

    net = ParamNet()
    res = net(Tensor([3], ms.float32))
    assert res == 0


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_1_2():
    """
    Feature: Check the names of parameters and the names of inputs of construct.
    Description: If the name of the input of construct is same as the parameters, add suffix to the name of the input.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = ParameterTuple((Parameter(Tensor([2], ms.float32), name="name_b"), self.param_a))

        def construct(self, name_b):
            return self.param_a + self.param_b[0] - name_b

    net = ParamNet()
    res = net(Tensor([3], ms.float32))
    assert res == 0


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_2_1():
    """
    Feature: Check the names of parameters.
    Description: If parameters in init have same name, an exception will be thrown.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = Parameter(Tensor([2], ms.float32), name="name_a")

        def construct(self):
            return self.param_a + self.param_b

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        res = net()
        assert res == 3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_2_2():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), self.param_a))
            self.param_a = Parameter(Tensor([3], ms.float32), name="name_a")
            self.res2 = self.res1[0] + self.param_a

        def construct(self):
            return self.param_a + self.res1[0] + self.res2

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        res = net()
        assert res == 10


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_3():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32))
            self.param_b = Parameter(Tensor([2], ms.float32))

        def construct(self):
            return self.param_a + self.param_b

    net = ParamNet()
    res = net()
    assert res == 3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_4():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32), name="name_a"),
                                        Parameter(Tensor([4], ms.float32), name="name_a")))

        def construct(self):
            return self.res1[0] + self.res1[1]

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        res = net()
        assert res == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_5_1():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), Parameter(Tensor([4], ms.float32))))

        def construct(self):
            return self.res1[0] + self.res1[1]

    with pytest.raises(ValueError, match="its name 'Parameter' already exists."):
        net = ParamNet()
        res = net()
        assert res == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_5_2():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32)), self.param_a))
            self.param_a = Parameter(Tensor([3], ms.float32), name="name_b")
            self.res2 = self.res1[0] + self.param_a

        def construct(self):
            return self.param_a + self.res1[0] + self.res2

    net = ParamNet()
    res = net()
    assert res == 10


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_list_tuple_no_name():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_tuple = (Parameter(Tensor([5], ms.float32)), Parameter(Tensor([6], ms.float32)))
            self.param_list = [Parameter(Tensor([7], ms.float32)), Parameter(Tensor([8], ms.float32))]

        def construct(self):
            return self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_list[1]

    net = ParamNet()
    res = net()
    assert res == 26


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_in_tuple():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = Parameter(Tensor([2], ms.float32), name="name_b")
            self.param_tuple = ParameterTuple((self.param_a, self.param_b))

        def construct(self):
            return self.param_a + self.param_b + self.param_tuple[0] + self.param_tuple[1]

    net = ParamNet()
    res = net()
    assert res == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_parameter_tuple_1():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_tuple = ParameterTuple((Parameter(Tensor([5], ms.float32), name="name_a"),
                                               Parameter(Tensor([5], ms.float32), name="name_b")))

        def construct(self):
            return self.param_a + self.param_tuple[0] + self.param_tuple[1]

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        res = net()
        assert res == 11


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parameter_parameter_tuple_2():
    """
    Feature: Check the names of parameters.
    Description: Check the name of parameter in init.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_tuple = ParameterTuple((self.param_a, self.param_a, self.param_a))

        def construct(self):
            return self.param_a + self.param_tuple[0] + self.param_tuple[1] + self.param_tuple[2]

    net = ParamNet()
    res = net()
    assert res == 4


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_parameter():
    """
    Feature: Check the names of parameters.
    Description: If parameter in list or tuple is not given a name, will give it a unique name.
    Expectation: No exception.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
            self.param_b = Parameter(Tensor([2], ms.float32), name="name_b")
            self.param_c = Parameter(Tensor([3], ms.float32))
            self.param_d = Parameter(Tensor([4], ms.float32))
            self.param_tuple = (Parameter(Tensor([5], ms.float32)),
                                Parameter(Tensor([6], ms.float32)))
            self.param_list = [Parameter(Tensor([5], ms.float32)),
                               Parameter(Tensor([6], ms.float32))]

        def construct(self, x):
            res1 = self.param_a + self.param_b + self.param_c + self.param_d
            res1 = res1 - self.param_list[0] + self.param_list[1] + x
            res2 = self.param_list[0] + self.param_list[1]
            return res1, res2

    net = ParamNet()
    x = Tensor([10], ms.float32)
    output1, output2 = net(x)
    output1_expect = Tensor(21, ms.float32)
    output2_expect = Tensor(11, ms.float32)
    assert output1 == output1_expect
    assert output2 == output2_expect


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_parameter_same_name_between_tuple_or_list():
    """
    Feature: Check the names of parameters between tuple or list.
    Description: If the same name exists between tuple and list, an exception will be thrown.
    Expectation: Get the expected exception report.
    """
    class ParamNet(Cell):
        def __init__(self):
            super(ParamNet, self).__init__()
            self.param_tuple = (Parameter(Tensor([1], ms.float32), name="name_a"),
                                Parameter(Tensor([2], ms.float32)))
            self.param_list = [Parameter(Tensor([3], ms.float32), name="name_a"),
                               Parameter(Tensor([4], ms.float32))]

        def construct(self, x):
            res = self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_listp[1] + x
            return res

    with pytest.raises(ValueError, match="its name 'name_a' already exists."):
        net = ParamNet()
        x = Tensor([10], ms.float32)
        output = net(x)
        output_expect = Tensor(20, ms.float32)
        assert output == output_expect
