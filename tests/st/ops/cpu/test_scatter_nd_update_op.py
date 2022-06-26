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

import numpy as np
import pytest

import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms import Parameter
from luojianet_ms.common import dtype as mstype
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op1(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(
                Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    update = Tensor(np.array([1.0, 2.2], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [[1.0, 0.3, 3.6], [0.4, 2.2, -3.2]]
    assert np.allclose(output.asnumpy(),
                       np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_op2(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(
                Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[4], [3], [1], [7]]), mstype.int32)
    update = Tensor(np.array([9, 10, 11, 12], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [1, 11, 3, 10, 9, 6, 7, 12]
    assert np.allclose(output.asnumpy(),
                       np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_op3(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.zeros((4, 4, 4)).astype(dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0], [2]]), mstype.int32)
    update = Tensor(np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]],
                              [[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    output = scatter_nd_update(indices, update)
    print("x:\n", output.asnumpy())
    expect = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    assert np.allclose(output.asnumpy(), np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op4(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate
    Expectation: success
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    update = Tensor(np.array([1.0], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    out = scatter_nd_update(indices, update)
    print("x:\n", out)
    expect = [[-0.1, 1.0, 3.6], [0.4, 0.5, -3.2]]
    assert np.allclose(out.asnumpy(), np.array(expect, dtype=dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_op5(dtype):
    """
    Feature: Op ScatterNdUpdate
    Description:  test ScatterNdUpdate with index out of range
    Expectation: raise RuntimeError
    """

    class ScatterNdUpdate(nn.Cell):
        def __init__(self):
            super(ScatterNdUpdate, self).__init__()
            self.scatter_nd_update = P.ScatterNdUpdate()
            self.x = Parameter(Tensor(np.ones([1, 4, 1], dtype=dtype)), name="x")

        def construct(self, indices, update):
            return self.scatter_nd_update(self.x, indices, update)

    indices = Tensor(np.array([[0, 2], [3, 2], [1, 3]]), mstype.int32)
    update = Tensor(np.array([[1], [1], [1]], dtype=dtype))

    scatter_nd_update = ScatterNdUpdate()
    with pytest.raises(RuntimeError) as errinfo:
        scatter_nd_update(indices, update)
    assert "Some errors occurred! The error message is as above" in str(errinfo.value)
