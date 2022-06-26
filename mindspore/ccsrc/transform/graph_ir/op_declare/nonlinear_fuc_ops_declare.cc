/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transform/graph_ir/op_declare/nonlinear_fuc_ops_declare.h"

namespace mindspore::transform {
// Relu
INPUT_MAP(Relu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Relu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Relu, prim::kPrimRelu->name(), ADPT_DESC(Relu))

// ReluV2
INPUT_MAP(ReluV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ReluV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ReluV2) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(ReluV2, kNameReluV2, ADPT_DESC(ReluV2))

// Elu
INPUT_MAP(Elu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Elu) = {{"alpha", ATTR_DESC(alpha, AnyTraits<float>())}};
OUTPUT_MAP(Elu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Elu, kNameElu, ADPT_DESC(Elu))

// EluGrad
INPUT_MAP(EluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(activations)}};
ATTR_MAP(EluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EluGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(EluGrad, kNameEluGrad, ADPT_DESC(EluGrad))

// PRelu
INPUT_MAP(PRelu) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(weight)}};
ATTR_MAP(PRelu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PRelu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(PRelu, kNamePrelu, ADPT_DESC(PRelu))

// PReluGrad
INPUT_MAP(PReluGrad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(features)}, {3, INPUT_DESC(weights)}};
ATTR_MAP(PReluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PReluGrad) = {{0, OUTPUT_DESC(dx)}, {1, OUTPUT_DESC(da)}};
REG_ADPT_DESC(PReluGrad, kNamePreluGrad, ADPT_DESC(PReluGrad))

// Selu
INPUT_MAP(Selu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Selu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Selu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Selu, kNameSelu, ADPT_DESC(Selu))

// Sigmoid
INPUT_MAP(Sigmoid) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sigmoid) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sigmoid) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Sigmoid, kNameSigmoid, ADPT_DESC(Sigmoid))

// SigmoidGrad
INPUT_MAP(SigmoidGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(SigmoidGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SigmoidGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(SigmoidGrad, kNameSigmoidGrad, ADPT_DESC(SigmoidGrad))

// HardSwish
INPUT_MAP(HardSwish) = {{1, INPUT_DESC(x)}};
ATTR_MAP(HardSwish) = EMPTY_ATTR_MAP;
OUTPUT_MAP(HardSwish) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(HardSwish, kNameHSwish, ADPT_DESC(HardSwish))

// HardSwishGrad
INPUT_MAP(HardSwishGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x)}};
ATTR_MAP(HardSwishGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(HardSwishGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(HardSwishGrad, kNameHSwishGrad, ADPT_DESC(HardSwishGrad))

// HSigmoid
INPUT_MAP(HardSigmoid) = {{1, INPUT_DESC(input_x)}};
ATTR_MAP(HardSigmoid) = {{"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                         {"beta", ATTR_DESC(beta, AnyTraits<float>())}};
OUTPUT_MAP(HardSigmoid) = {{0, OUTPUT_DESC(output_y)}};
REG_ADPT_DESC(HardSigmoid, kNameHSigmoid, ADPT_DESC(HardSigmoid))

// Relu6
INPUT_MAP(Relu6) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Relu6) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu6) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Relu6, kNameReLU6, ADPT_DESC(Relu6))

// Relu6Grad
INPUT_MAP(Relu6Grad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
ATTR_MAP(Relu6Grad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Relu6Grad) = {{0, OUTPUT_DESC(backprops)}};
REG_ADPT_DESC(Relu6Grad, kNameReLU6Grad, ADPT_DESC(Relu6Grad))

// Softsign
INPUT_MAP(Softsign) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Softsign) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Softsign) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Softsign, kNameSoftsign, ADPT_DESC(Softsign))

// Softplus
INPUT_MAP(Softplus) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Softplus) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Softplus) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Softplus, kNameSoftplus, ADPT_DESC(Softplus))

// SoftplusGrad
INPUT_MAP(SoftplusGrad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
ATTR_MAP(SoftplusGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SoftplusGrad) = {{0, OUTPUT_DESC(backprops)}};
REG_ADPT_DESC(SoftplusGrad, kNameSoftplusGrad, ADPT_DESC(SoftplusGrad))

// ReluGrad
INPUT_MAP(ReluGrad) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(features)}};
ATTR_MAP(ReluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ReluGrad) = {{0, OUTPUT_DESC(backprops)}};
REG_ADPT_DESC(ReluGrad, prim::kPrimReluGrad->name(), ADPT_DESC(ReluGrad))

// ReluGradV2
INPUT_MAP(ReluGradV2) = {{1, INPUT_DESC(gradients)}, {2, INPUT_DESC(mask)}};
ATTR_MAP(ReluGradV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ReluGradV2) = {{0, OUTPUT_DESC(backprops)}};
REG_ADPT_DESC(ReluGradV2, kNameReluGradV2, ADPT_DESC(ReluGradV2))

// Tanh
INPUT_MAP(Tanh) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Tanh) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Tanh) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Tanh, prim::kPrimTanh->name(), ADPT_DESC(Tanh))

// TanhGrad
INPUT_MAP(TanhGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(TanhGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TanhGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(TanhGrad, prim::kPrimTanhGrad->name(), ADPT_DESC(TanhGrad))

// Mish
INPUT_MAP(Mish) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Mish) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Mish) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Mish, kNameMish, ADPT_DESC(Mish))

// GeLU
INPUT_MAP(Gelu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Gelu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Gelu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Gelu, prim::kPrimGeLU->name(), ADPT_DESC(Gelu))

// GeLUGrad
INPUT_MAP(GeluGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(y)}};
ATTR_MAP(GeluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GeluGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(GeluGrad, prim::kPrimGeLUGrad->name(), ADPT_DESC(GeluGrad))

// FastGeLU
INPUT_MAP(FastGelu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(FastGelu) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FastGelu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FastGelu, prim::kPrimFastGeLU->name(), ADPT_DESC(FastGelu))

// FastGeLUGrad
INPUT_MAP(FastGeluGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}};
ATTR_MAP(FastGeluGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FastGeluGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(FastGeluGrad, prim::kPrimFastGeLUGrad->name(), ADPT_DESC(FastGeluGrad))

// LeakyRelu
INPUT_MAP(LeakyRelu) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LeakyRelu) = {{"alpha", ATTR_DESC(negative_slope, AnyTraits<float>())}};
OUTPUT_MAP(LeakyRelu) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LeakyRelu, prim::kPrimLeakyRelu->name(), ADPT_DESC(LeakyRelu))
}  // namespace mindspore::transform
