# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn.probability.distribution as msd

context.set_context(device_target='GPU')

def test_categorical1():
    cat1 = msd.Categorical(probs=[[0.9, 0.2], [0.9, 0.2]])
    cat1out1 = cat1.sample((1,))
    cat1out2 = cat1.sample((3, 2))
    cat1out3 = cat1.sample((6,))
    assert cat1out1.asnumpy().shape == (2, 1)
    assert cat1out2.asnumpy().shape == (2, 3, 2)
    assert cat1out3.asnumpy().shape == (2, 6)

    cat1 = msd.Categorical(probs=[0.9, 0.2])
    cat1out1 = cat1.sample((1,))
    cat1out2 = cat1.sample((3, 2))
    cat1out3 = cat1.sample((6,))
    assert cat1out1.asnumpy().shape == (1,)
    assert cat1out2.asnumpy().shape == (3, 2)
    assert cat1out3.asnumpy().shape == (6,)
