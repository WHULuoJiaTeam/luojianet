/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"
#include "nnacl/arithmetic_self_parameter.h"

namespace mindspore::lite::micro::nnacl {
using mindspore::schema::PrimitiveType_Abs;

using mindspore::schema::PrimitiveType_AddFusion;

using mindspore::schema::PrimitiveType_AddN;

using mindspore::schema::PrimitiveType_Neg;

using mindspore::schema::PrimitiveType_Ceil;

using mindspore::schema::PrimitiveType_Cos;

using mindspore::schema::PrimitiveType_DivFusion;

using mindspore::schema::PrimitiveType_Equal;

using mindspore::schema::PrimitiveType_Floor;

using mindspore::schema::PrimitiveType_FloorDiv;

using mindspore::schema::PrimitiveType_FloorMod;

using mindspore::schema::PrimitiveType_Greater;

using mindspore::schema::PrimitiveType_GreaterEqual;

using mindspore::schema::PrimitiveType_Less;

using mindspore::schema::PrimitiveType_LessEqual;

using mindspore::schema::PrimitiveType_Log;

using mindspore::schema::PrimitiveType_LogicalAnd;

using mindspore::schema::PrimitiveType_LogicalOr;

using mindspore::schema::PrimitiveType_LogicalNot;

using mindspore::schema::PrimitiveType_Maximum;

using mindspore::schema::PrimitiveType_Minimum;

using mindspore::schema::PrimitiveType_MulFusion;

using mindspore::schema::PrimitiveType_NotEqual;

using mindspore::schema::PrimitiveType_RealDiv;

using mindspore::schema::PrimitiveType_Round;

using mindspore::schema::PrimitiveType_Rsqrt;

using mindspore::schema::PrimitiveType_Sqrt;

using mindspore::schema::PrimitiveType_SquaredDifference;

using mindspore::schema::PrimitiveType_SubFusion;

using mindspore::schema::PrimitiveType_Sin;

using mindspore::schema::PrimitiveType_Square;

using mindspore::schema::PrimitiveType_Erf;

class ArithmeticSelfFP32Coder final : public OperatorCoder {
 public:
  ArithmeticSelfFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                          const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;
  ~ArithmeticSelfFP32Coder() override = default;

 private:
  int ReSize();

 private:
  int thread_sz_count_{0};
  int thread_sz_stride_{0};
  size_t data_size_{0};
  std::string arithmetic_self_run_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_
