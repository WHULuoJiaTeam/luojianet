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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ARITHMETIC_GRAD_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ARITHMETIC_GRAD_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "schema/model_generated.h"

using luojianet_ms::schema::PrimitiveType_AddGrad;
using luojianet_ms::schema::PrimitiveType_DivGrad;
using luojianet_ms::schema::PrimitiveType_MaximumGrad;
using luojianet_ms::schema::PrimitiveType_MinimumGrad;
using luojianet_ms::schema::PrimitiveType_MulGrad;
using luojianet_ms::schema::PrimitiveType_SubGrad;

namespace luojianet_ms::kernel {

class ArithmeticGradCPUKernel;

class ArithmeticGradCPUKernel : public InnerKernel {
  typedef int (ArithmeticGradCPUKernel::*ArithmeticGradOperation)(float *, int, float *, int, float *, int);

 public:
  explicit ArithmeticGradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx),
        arithmetic_grad_(nullptr),
        tile_data0(nullptr),
        tile_data1(nullptr),
        tile_data2(nullptr) {
    switch (parameter->type_) {
      case PrimitiveType_MulGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMul;  // this will be adjusted in InferShape
        break;
      case PrimitiveType_AddGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradAdd;
        break;
      case PrimitiveType_SubGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradSub;
        break;
      case PrimitiveType_DivGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradDiv;  // this will be adjusted in InferShape
        break;
      case PrimitiveType_MaximumGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMaximum;
        break;
      case PrimitiveType_MinimumGrad:
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMinimum;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << parameter->type_;
        break;
    }
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~ArithmeticGradCPUKernel() override {
    if (tile_data0 != nullptr) delete[] tile_data0;
    if (tile_data1 != nullptr) delete[] tile_data1;
    if (tile_data2 != nullptr) delete[] tile_data2;
  }

  int Prepare() override;
  int InferShape();
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int ArithmeticGradAdd(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradSub(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradMul(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradMul1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradMul2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradDiv(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradDiv1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradDiv2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradMaximum(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  int ArithmeticGradMinimum(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2, int dx2_size);
  ArithmeticParameter *arithmeticParameter_;
  ArithmeticGradOperation arithmetic_grad_;
  float *tile_data0;
  float *tile_data1;
  float *tile_data2;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ARITHMETIC_GRAD_H_
