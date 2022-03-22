/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"
#include "nnacl/arithmetic_self_parameter.h"

namespace luojianet_ms::lite::micro::nnacl {

using luojianet_ms::schema::PrimitiveType_Abs;

using luojianet_ms::schema::PrimitiveType_AddFusion;

using luojianet_ms::schema::PrimitiveType_AddN;

using luojianet_ms::schema::PrimitiveType_Neg;

using luojianet_ms::schema::PrimitiveType_Ceil;

using luojianet_ms::schema::PrimitiveType_Cos;

using luojianet_ms::schema::PrimitiveType_DivFusion;

using luojianet_ms::schema::PrimitiveType_Equal;

using luojianet_ms::schema::PrimitiveType_Floor;

using luojianet_ms::schema::PrimitiveType_FloorDiv;

using luojianet_ms::schema::PrimitiveType_FloorMod;

using luojianet_ms::schema::PrimitiveType_Greater;

using luojianet_ms::schema::PrimitiveType_GreaterEqual;

using luojianet_ms::schema::PrimitiveType_Less;

using luojianet_ms::schema::PrimitiveType_LessEqual;

using luojianet_ms::schema::PrimitiveType_Log;

using luojianet_ms::schema::PrimitiveType_LogicalAnd;

using luojianet_ms::schema::PrimitiveType_LogicalOr;

using luojianet_ms::schema::PrimitiveType_LogicalNot;

using luojianet_ms::schema::PrimitiveType_Maximum;

using luojianet_ms::schema::PrimitiveType_Minimum;

using luojianet_ms::schema::PrimitiveType_MulFusion;

using luojianet_ms::schema::PrimitiveType_NotEqual;

using luojianet_ms::schema::PrimitiveType_RealDiv;

using luojianet_ms::schema::PrimitiveType_Round;

using luojianet_ms::schema::PrimitiveType_Rsqrt;

using luojianet_ms::schema::PrimitiveType_Sqrt;

using luojianet_ms::schema::PrimitiveType_SquaredDifference;

using luojianet_ms::schema::PrimitiveType_SubFusion;

using luojianet_ms::schema::PrimitiveType_Sin;

using luojianet_ms::schema::PrimitiveType_Square;

using luojianet_ms::schema::PrimitiveType_Erf;

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
}  // namespace luojianet_ms::lite::micro::nnacl
#endif  // LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_SELF_FP32_CODER_H_
