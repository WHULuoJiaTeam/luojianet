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
import pytest
import numpy as np
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scatter = P.TensorScatterUpdate()

    def forward(self, x, indices, update):
        return self.scatter(x, indices, update)


def scatter_net(x, indices, update):
    scatter = Net()
    return scatter(Tensor(x), Tensor(indices), Tensor(update)).asnumpy()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    arr_input = np.arange(21).reshape(3, 7).astype(np.float32)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(np.int32)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[0, 3.2, -2.2, 3, 4, 5.3, 6],
                         [7, 1.1, 9, 10, 11, 12, 13],
                         [14, -1, 16, 17, 18, 19, 20]]).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(24).reshape(4, 2, 3).astype(np.float32)
    arr_indices = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1], [3, 0, 1]]).astype(np.int32)
    arr_update = np.array([-1, -2, -3, -4]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[[-1, 1, 2],
                          [3, -3, 5]],
                         [[6, 7, 8],
                          [9, -2, 11]],
                         [[12, 13, 14],
                          [15, 16, 17]],
                         [[18, -4, 20],
                          [21, 22, 23]]]).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(25).reshape(5, 5).astype(np.float32)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int32)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.float32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 66],
                         [5, 22, 7, 77, 9],
                         [10, 11, 33, 13, 14],
                         [15, 99, 17, 44, 19],
                         [100, 21, 22, 23, 55]]).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(25).reshape(5, 5).astype(np.float64)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int64)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.float64)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 66],
                         [5, 22, 7, 77, 9],
                         [10, 11, 33, 13, 14],
                         [15, 99, 17, 44, 19],
                         [100, 21, 22, 23, 55]]).astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-6)

    arr_input = np.arange(25).reshape(5, 5).astype(np.int32)
    arr_indices = np.array([[[0, 0],
                             [1, 1],
                             [2, 2],
                             [3, 3],
                             [4, 4]],
                            [[0, 4],
                             [1, 3],
                             [2, 2],
                             [3, 1],
                             [4, 0]]]).astype(np.int32)
    arr_update = np.array([[11, 22, 33, 44, 55], [66, 77, 33, 99, 100]]).astype(np.int32)
    out = scatter_net(arr_input, arr_indices, arr_update)
    expected = np.array([[11, 1, 2, 3, 66],
                         [5, 22, 7, 77, 9],
                         [10, 11, 33, 13, 14],
                         [15, 99, 17, 44, 19],
                         [100, 21, 22, 23, 55]]).astype(np.int32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
