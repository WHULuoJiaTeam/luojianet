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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_SELF_PARAMETER_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_SELF_PARAMETER_H_

#include <vector>
#include <string>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/arithmetic_self_parameter.h"

using luojianet_ms::schema::PrimitiveType_Abs;
using luojianet_ms::schema::PrimitiveType_Ceil;
using luojianet_ms::schema::PrimitiveType_Cos;
using luojianet_ms::schema::PrimitiveType_Eltwise;
using luojianet_ms::schema::PrimitiveType_ExpFusion;
using luojianet_ms::schema::PrimitiveType_Floor;
using luojianet_ms::schema::PrimitiveType_Log;
using luojianet_ms::schema::PrimitiveType_LogicalNot;
using luojianet_ms::schema::PrimitiveType_Neg;
using luojianet_ms::schema::PrimitiveType_Round;
using luojianet_ms::schema::PrimitiveType_Rsqrt;
using luojianet_ms::schema::PrimitiveType_Sin;
using luojianet_ms::schema::PrimitiveType_Sqrt;
using luojianet_ms::schema::PrimitiveType_Square;

namespace luojianet_ms::kernel {
class ArithmeticSelfOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~ArithmeticSelfOpenCLKernel() override = default;

  int Prepare() override;

  int CheckSpecs() override;
  int CheckSpecsWithoutShape() override;
  int SetConstArgs() override {
    if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX2, output_shape_) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  int SetGlobalLocal() override;

  int Run() override;

 private:
  cl_int4 output_shape_ = {};
};

}  // namespace luojianet_ms::kernel
#endif
