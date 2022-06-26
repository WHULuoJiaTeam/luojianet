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
#include "src/runtime/kernel/arm/fp32/arithmetic_self_fp32.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"

using luojianet_ms::lite::KernelRegistrar;
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;

namespace luojianet_ms::kernel {
namespace {
struct TYPE_FUNC_INFO {
  int primitive_type_ = 0;
  ArithmeticSelfFunc func_ = nullptr;
};
}  // namespace

ArithmeticSelfFunc ArithmeticSelfCPUKernel::GetArithmeticSelfFun(int primitive_type) const {
  TYPE_FUNC_INFO type_func_table[] = {{luojianet_ms::schema::PrimitiveType_Abs, ElementAbs},
                                      {luojianet_ms::schema::PrimitiveType_Cos, ElementCos},
                                      {luojianet_ms::schema::PrimitiveType_Log, ElementLog},
                                      {luojianet_ms::schema::PrimitiveType_Square, ElementSquare},
                                      {luojianet_ms::schema::PrimitiveType_Sqrt, ElementSqrt},
                                      {luojianet_ms::schema::PrimitiveType_Rsqrt, ElementRsqrt},
                                      {luojianet_ms::schema::PrimitiveType_Sin, ElementSin},
                                      {luojianet_ms::schema::PrimitiveType_LogicalNot, ElementLogicalNot},
                                      {luojianet_ms::schema::PrimitiveType_Floor, ElementFloor},
                                      {luojianet_ms::schema::PrimitiveType_Ceil, ElementCeil},
                                      {luojianet_ms::schema::PrimitiveType_Round, ElementRound},
                                      {luojianet_ms::schema::PrimitiveType_Neg, ElementNegative},
                                      {luojianet_ms::schema::PrimitiveType_Reciprocal, ElementReciprocal},
                                      {luojianet_ms::schema::PrimitiveType_Erf, ElementErf}};
  for (size_t i = 0; i < sizeof(type_func_table) / sizeof(TYPE_FUNC_INFO); i++) {
    if (type_func_table[i].primitive_type_ == primitive_type) {
      return type_func_table[i].func_;
    }
  }
  return nullptr;
}

ArithmeticSelfBoolFunc ArithmeticSelfCPUKernel::GetArithmeticSelfBoolFun(int primitive_type) const {
  if (primitive_type == luojianet_ms::schema::PrimitiveType_LogicalNot) {
    return ElementLogicalNotBool;
  }
  return nullptr;
}

int ArithmeticSelfCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticSelfCPUKernel::ReSize() {
  if (UpdateThreadNumPass(TC_PTYPE(type_), 1, 1, out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfCPUKernel::DoExecute(int task_id) {
  int elements_num = in_tensors_.at(0)->ElementsNum();
  MS_CHECK_TRUE_RET(thread_num_ != 0, RET_ERROR);
  int stride = UP_DIV(elements_num, thread_num_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, stride, RET_ERROR);
  int offset = task_id * stride;
  int count = MSMIN(stride, elements_num - offset);
  if (count <= 0) {
    return RET_OK;
  }
  int ret = RET_ERROR;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (func_ == nullptr) {
      MS_LOG(ERROR) << "Run function is null! ";
      return RET_ERROR;
    }
    float *input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data());
    float *output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
    ret = func_(input_ptr + offset, output_ptr + offset, count);
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    if (func_bool_ == nullptr) {
      MS_LOG(ERROR) << "Run function is null! ";
      return RET_ERROR;
    }
    bool *input_ptr = reinterpret_cast<bool *>(in_tensors_.at(0)->data());
    bool *output_ptr = reinterpret_cast<bool *>(out_tensors_.at(0)->data());
    ret = func_bool_(input_ptr + offset, output_ptr + offset, count);
  } else {
    MS_LOG(ERROR) << "Unsupported type: " << in_tensors_[0]->data_type() << ".";
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run failed, illegal input! ";
  }
  return ret;
}

int ArithmeticSelfRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<ArithmeticSelfCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRuns error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticSelfCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, ArithmeticSelfRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticSelfRun error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Abs, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cos, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Log, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Square, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sqrt, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sin, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalNot, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Floor, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Ceil, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Round, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Neg, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reciprocal, LiteKernelCreator<ArithmeticSelfCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Erf, LiteKernelCreator<ArithmeticSelfCPUKernel>)
}  // namespace luojianet_ms::kernel
