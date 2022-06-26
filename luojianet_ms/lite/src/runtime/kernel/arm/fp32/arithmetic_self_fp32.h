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
#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_

#include <vector>
#include <map>
#include <memory>
#include "src/inner_kernel.h"

using luojianet_ms::schema::PrimitiveType_Abs;
using luojianet_ms::schema::PrimitiveType_Ceil;
using luojianet_ms::schema::PrimitiveType_Cos;
using luojianet_ms::schema::PrimitiveType_Erf;
using luojianet_ms::schema::PrimitiveType_Floor;
using luojianet_ms::schema::PrimitiveType_Log;
using luojianet_ms::schema::PrimitiveType_LogicalNot;
using luojianet_ms::schema::PrimitiveType_Neg;
using luojianet_ms::schema::PrimitiveType_Reciprocal;
using luojianet_ms::schema::PrimitiveType_Round;
using luojianet_ms::schema::PrimitiveType_Rsqrt;
using luojianet_ms::schema::PrimitiveType_Sin;
using luojianet_ms::schema::PrimitiveType_Sqrt;
using luojianet_ms::schema::PrimitiveType_Square;

namespace luojianet_ms::kernel {
typedef int (*ArithmeticSelfFunc)(const float *input, float *output, const int element_size);
typedef int (*ArithmeticSelfBoolFunc)(const bool *input, bool *output, const int element_size);
class ArithmeticSelfCPUKernel : public InnerKernel {
 public:
  explicit ArithmeticSelfCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    func_ = GetArithmeticSelfFun(parameter->type_);
    func_bool_ = GetArithmeticSelfBoolFun(parameter->type_);
  }
  ~ArithmeticSelfCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int DoExecute(int task_id);

 private:
  ArithmeticSelfFunc GetArithmeticSelfFun(int primitive_type) const;
  ArithmeticSelfBoolFunc GetArithmeticSelfBoolFun(int primitive_type) const;
  ArithmeticSelfFunc func_;
  ArithmeticSelfBoolFunc func_bool_;
};
int ArithmeticSelfRun(void *cdata, int task_id, float lhs_scale, float rhs_scale);
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ARITHMETIC_SELF_H_
