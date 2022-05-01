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

import luojianet_ms as ms
import luojianet_ms.context as context
from luojianet_ms import Tensor
from luojianet_ms.nn import Module
from luojianet_ms.ops import operations as P
from luojianet_ms.train.model import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class PowMe(Module):
    def __init__(self):
        super(PowMe, self).__init__()
        self.pow = P.Pow()

    def call(self, input_, exp):
        return self.pow(input_, exp)


def pow_forward_me_impl(input_, exp):
    n = PowMe()
    n.set_train()
    m = Model(n)
    out = m.predict(input_, exp)
    return out.asnumpy()


def pow_forward_cmp(input_shape, exp_shape):
    if not input_shape:
        input_np = np.absolute(np.random.randn())
    else:
        input_np = np.absolute(np.random.randn(*input_shape).astype(np.float32))
    input_me = Tensor(input_np, dtype=ms.float32)

    if not exp_shape:
        exp_np = np.absolute(np.random.randn())
    else:
        exp_np = np.absolute(np.random.randn(*exp_shape).astype(np.float32))
    exp_me = Tensor(exp_np, dtype=ms.float32)

    out_me = pow_forward_me_impl(input_me, exp_me)
    print(input_me)
    print(exp_me)
    print(out_me)


def test_pow_input_scalar_exp_scalar():
    input_shape = []
    exp_shape = []
    pow_forward_cmp(input_shape, exp_shape)
