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
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
BatchNorm = P.BatchNorm()
BNTrainingReduce = Primitive('BNTrainingReduce')
BNTrainingUpdateV2 = Primitive('BNTrainingUpdateV2')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_batch_norm_bert_fission(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4):
        batch_norm = BatchNorm(input0, input1, input2, input3, input4)
        outputs = make_tuple(tuple_getitem(batch_norm, 0), tuple_getitem(batch_norm, 3), tuple_getitem(batch_norm, 4))
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4):
        bn_training_reduce = BNTrainingReduce(input0)
        bn_training_update_v2 = BNTrainingUpdateV2(input0, tuple_getitem(bn_training_reduce, 0),
                                                   tuple_getitem(bn_training_reduce, 1), input1, input2)
        outputs = make_tuple(tuple_getitem(bn_training_update_v2, 0), tuple_getitem(bn_training_update_v2, 1),
                             tuple_getitem(bn_training_update_v2, 2))
        return make_tuple(outputs)

    return fns[tag]
