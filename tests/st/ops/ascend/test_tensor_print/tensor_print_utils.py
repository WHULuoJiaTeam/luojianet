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

import luojianet_ms
import luojianet_ms.context as context
import luojianet_ms.nn as nn
from luojianet_ms import Tensor
from luojianet_ms.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class TensorPrint(nn.Module):
    def __init__(self):
        super().__init__()
        self.print = P.Print()

    def forward(self, *inputs):
        self.print(*inputs)
        return inputs[0]

def get_tensor(is_scalar, input_type):
    if is_scalar == 'scalar':
        if input_type == luojianet_ms.bool_:
            return Tensor(True, dtype=input_type)
        if input_type in [luojianet_ms.uint8, luojianet_ms.uint16, luojianet_ms.uint32, luojianet_ms.uint64]:
            return Tensor(1, dtype=input_type)
        if input_type in [luojianet_ms.int8, luojianet_ms.int16, luojianet_ms.int32, luojianet_ms.int64]:
            return Tensor(-1, dtype=input_type)
        if input_type in [luojianet_ms.float16, luojianet_ms.float32, luojianet_ms.float64]:
            return Tensor(0.01, dtype=input_type)
    else:
        if input_type == luojianet_ms.bool_:
            return Tensor(np.array([[True, False], [False, True]]), dtype=input_type)
        if input_type in [luojianet_ms.uint8, luojianet_ms.uint16, luojianet_ms.uint32, luojianet_ms.uint64]:
            return Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=input_type)
        if input_type in [luojianet_ms.int8, luojianet_ms.int16, luojianet_ms.int32, luojianet_ms.int64]:
            return Tensor(np.array([[-1, 2, -3], [-4, 5, -6]]), dtype=input_type)
        if input_type in [luojianet_ms.float16, luojianet_ms.float32, luojianet_ms.float64]:
            return Tensor(np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]), dtype=input_type)
    return Tensor(False, np.bool)

if __name__ == "__main__":
    net = TensorPrint()
    net(get_tensor('scalar', luojianet_ms.bool_), get_tensor('scalar', luojianet_ms.uint8),
        get_tensor('scalar', luojianet_ms.int8), get_tensor('scalar', luojianet_ms.uint16),
        get_tensor('scalar', luojianet_ms.int16), get_tensor('scalar', luojianet_ms.uint32),
        get_tensor('scalar', luojianet_ms.int32), get_tensor('scalar', luojianet_ms.uint64),
        get_tensor('scalar', luojianet_ms.int64), get_tensor('scalar', luojianet_ms.float16),
        get_tensor('scalar', luojianet_ms.float32), get_tensor('scalar', luojianet_ms.float64),
        get_tensor('array', luojianet_ms.bool_), get_tensor('array', luojianet_ms.uint8),
        get_tensor('array', luojianet_ms.int8), get_tensor('array', luojianet_ms.uint16),
        get_tensor('array', luojianet_ms.int16), get_tensor('array', luojianet_ms.uint32),
        get_tensor('array', luojianet_ms.int32), get_tensor('array', luojianet_ms.uint64),
        get_tensor('array', luojianet_ms.int64), get_tensor('array', luojianet_ms.float16),
        get_tensor('array', luojianet_ms.float32), get_tensor('array', luojianet_ms.float64))
