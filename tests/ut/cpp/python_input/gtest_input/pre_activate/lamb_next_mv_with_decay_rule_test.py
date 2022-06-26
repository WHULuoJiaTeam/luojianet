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

Add = P.Add()
Mul = P.Mul()
RealDiv = P.RealDiv()
Rsqrt = P.Rsqrt()
Sqrt = P.Sqrt()
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
LambNextMVWithDecay = Primitive('LambNextMVWithDecay')

class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]

def test_lamb_next_mv_with_decay_rule_cond4(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
               constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(real_div0, sqrt0)
        add5 = Add(real_div4, mul4)
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        lamb_next_mv_with_decay = LambNextMVWithDecay(input0, input1, input2, input3, input4, input5, input6,
                                                      constant_mul0_x, constant_mul1_sub,
                                                      constant_mul2_x, constant_mul3_sub1, constant_mul4_x,
                                                      constant_add2_y)
        outputs = make_tuple(tuple_getitem(lamb_next_mv_with_decay, 0), tuple_getitem(lamb_next_mv_with_decay, 1),
                             tuple_getitem(lamb_next_mv_with_decay, 2), tuple_getitem(lamb_next_mv_with_decay, 3))
        return make_tuple(outputs)

    @fns
    def before_unmatched_add3(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x,
                              constant_mul1_sub,
                              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(real_div0, sqrt0)
        add5 = Add(real_div4, mul4)
        add3 = Mul(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def before_unmatched_mul4(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x,
                              constant_mul1_sub,
                              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Add(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(real_div0, sqrt0)
        add5 = Add(real_div4, mul4)
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def before_unmatched_real_div0(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x,
                                   constant_mul1_sub,
                                   constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = Add(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(real_div0, sqrt0)
        add5 = Add(real_div4, mul4)
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def before_unmatched_real_div1(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x,
                                   constant_mul1_sub,
                                   constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = Add(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(real_div0, sqrt0)
        add5 = Add(real_div4, mul4)
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    return fns[tag]

def test_lamb_next_mv_with_decay_rule_cond1(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
               constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(input3, constant_mul1_sub)
        mul0 = Mul(input4, constant_mul0_x)
        add0 = Add(mul0, mul1)
        mul2 = Mul(input1, constant_mul2_x)
        mul3 = Mul(input0, constant_mul3_sub1)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(constant_add2_y, real_div1)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        add3 = Add(mul4, real_div2)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        lamb_next_mv_with_decay = LambNextMVWithDecay(input0, input1, input2, input3, input4, input5, input6,
                                                      constant_mul0_x, constant_mul1_sub,
                                                      constant_mul2_x, constant_mul3_sub1, constant_mul4_x,
                                                      constant_add2_y)
        outputs = make_tuple(tuple_getitem(lamb_next_mv_with_decay, 0), tuple_getitem(lamb_next_mv_with_decay, 1),
                             tuple_getitem(lamb_next_mv_with_decay, 2), tuple_getitem(lamb_next_mv_with_decay, 3))
        return make_tuple(outputs)

    @fns
    def un_match(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
                 constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(input3, constant_mul1_sub)
        mul0 = Mul(input4, constant_mul0_x)
        add0 = Add(mul0, mul1)
        mul2 = Mul(input1, constant_mul2_x)
        mul3 = Mul(input0, constant_mul3_sub1)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(constant_add2_y, real_div1)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        # un match
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    return fns[tag]

def test_lamb_next_mv_with_decay_rule_cond2(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
               constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(constant_add2_y, real_div1)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(constant_add2_y, sqrt1)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        add3 = Add(mul4, real_div2)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        lamb_next_mv_with_decay = LambNextMVWithDecay(input0, input1, input2, input3, input4, input5, input6,
                                                      constant_mul0_x, constant_mul1_sub,
                                                      constant_mul2_x, constant_mul3_sub1, constant_mul4_x,
                                                      constant_add2_y)
        outputs = make_tuple(tuple_getitem(lamb_next_mv_with_decay, 0), tuple_getitem(lamb_next_mv_with_decay, 1),
                             tuple_getitem(lamb_next_mv_with_decay, 2), tuple_getitem(lamb_next_mv_with_decay, 3))
        return make_tuple(outputs)

    @fns
    def un_match(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
                 constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(constant_mul1_sub, input3)
        mul0 = Mul(constant_mul0_x, input4)
        add0 = Add(mul0, mul1)
        mul2 = Mul(constant_mul2_x, input1)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(constant_add2_y, real_div1)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(constant_add2_y, sqrt1)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(constant_mul4_x, input6)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        # un_match
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    return fns[tag]

def test_lamb_next_mv_with_decay_rule_cond3(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
               constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(input3, constant_mul1_sub)
        mul0 = Mul(input4, constant_mul0_x)
        add0 = Add(mul0, mul1)
        mul2 = Mul(input1, constant_mul2_x)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(input6, constant_mul4_x)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        add3 = Add(mul4, real_div2)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    @fns
    def after(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
              constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        lamb_next_mv_with_decay = LambNextMVWithDecay(input0, input1, input2, input3, input4, input5, input6,
                                                      constant_mul0_x, constant_mul1_sub,
                                                      constant_mul2_x, constant_mul3_sub1, constant_mul4_x,
                                                      constant_add2_y)
        outputs = make_tuple(tuple_getitem(lamb_next_mv_with_decay, 0), tuple_getitem(lamb_next_mv_with_decay, 1),
                             tuple_getitem(lamb_next_mv_with_decay, 2), tuple_getitem(lamb_next_mv_with_decay, 3))
        return make_tuple(outputs)

    @fns
    def un_match(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
                 constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
        mul1 = Mul(input3, constant_mul1_sub)
        mul0 = Mul(input4, constant_mul0_x)
        add0 = Add(mul0, mul1)
        mul2 = Mul(input1, constant_mul2_x)
        mul3 = Mul(constant_mul3_sub1, input0)
        add1 = Add(mul2, mul3)
        real_div1 = RealDiv(add1, input2)
        add2 = Add(real_div1, constant_add2_y)
        sqrt1 = Sqrt(real_div1)
        real_div0 = RealDiv(add0, input5)
        add4 = Add(sqrt1, constant_add2_y)
        sqrt0 = Rsqrt(add2)
        mul4 = Mul(input6, constant_mul4_x)
        real_div4 = RealDiv(real_div0, add4)
        real_div2 = Mul(sqrt0, real_div0)
        add5 = Add(mul4, real_div4)
        # un match
        add3 = Add(real_div2, mul4)
        outputs = make_tuple(add3, add0, add1, add5)
        return outputs

    return fns[tag]
