/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/int8/arithmetic_int8.h"
#include "src/runtime/kernel/arm/int8/add_int8.h"
#include "src/runtime/kernel/arm/int8/mul_int8.h"
#include "nnacl/arithmetic.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;

using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_NotEqual;

namespace mindspore::kernel {
namespace {
int ArithmeticsInt8Launch(void *cdata, int task_id, float, float) {
  auto arithmetic_kernel = reinterpret_cast<ArithmeticInt8CPUKernel *>(cdata);
  auto error_code = arithmetic_kernel->DoArithmetic(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error thread_id[" << task_id << "] error_code[" << error_code << "]";
    return error_code;
  }
  return RET_OK;
}
}  // namespace

int ArithmeticInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  switch (op_parameter_->type_) {
    case PrimitiveType_Equal:
      arithmetic_run_ = ElementEqualInt8;
      break;
    case PrimitiveType_NotEqual:
      arithmetic_run_ = ElementNotEqualInt8;
      break;
    case PrimitiveType_Less:
      arithmetic_run_ = ElementLessInt8;
      break;
    case PrimitiveType_LessEqual:
      arithmetic_run_ = ElementLessEqualInt8;
      break;
    case PrimitiveType_Greater:
      arithmetic_run_ = ElementGreaterInt8;
      break;
    case PrimitiveType_GreaterEqual:
      arithmetic_run_ = ElementGreaterEqualInt8;
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
      arithmetic_run_ = nullptr;
      return RET_PARAM_INVALID;
  }

  auto *input0_tensor = in_tensors_.at(0);
  auto in0_quant_args = input0_tensor->quant_params();
  CHECK_LESS_RETURN(in0_quant_args.size(), 1);
  quant_args_.in0_args_.scale_ = in0_quant_args.front().scale;
  quant_args_.in0_args_.zp_ = in0_quant_args.front().zeroPoint;

  auto *input1_tensor = in_tensors_.at(1);
  auto in1_quant_args = input1_tensor->quant_params();
  CHECK_LESS_RETURN(in1_quant_args.size(), 1);
  quant_args_.in1_args_.scale_ = in1_quant_args.front().scale;
  quant_args_.in1_args_.zp_ = in1_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  quant_args_.out_args_.scale_ = out_quant_args.front().scale;
  quant_args_.out_args_.zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int ArithmeticInt8CPUKernel::ReSize() { return RET_OK; }

int ArithmeticInt8CPUKernel::DoArithmetic(int thread_id) {
  auto input0_data = reinterpret_cast<int8_t *>(in_tensors_[0]->MutableData());
  CHECK_NULL_RETURN(input0_data);
  auto input1_data = reinterpret_cast<int8_t *>(in_tensors_[1]->MutableData());
  CHECK_NULL_RETURN(input1_data);
  auto output_data = reinterpret_cast<uint8_t *>(out_tensors_[0]->MutableData());
  CHECK_NULL_RETURN(output_data);
  auto element_num = out_tensors_[0]->ElementsNum();
  MS_CHECK_GT(element_num, 0, RET_ERROR);
  auto param = reinterpret_cast<ArithmeticParameter *>(op_parameter_);
  int error_code;
  if (param->broadcasting_ && arithmetic_run_ != nullptr) {
    MS_ASSERT(op_parameter_->thread_num_ != 0);
    int stride = UP_DIV(element_num, op_parameter_->thread_num_);
    int count = MSMIN(stride, element_num - stride * thread_id);
    if (count <= 0) {
      return RET_OK;
    }

    error_code = arithmetic_run_(tile_data0_ + stride * thread_id, tile_data1_ + stride * thread_id,
                                 output_data + stride * thread_id, count, &quant_args_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Arithmetic run fail! ret: " << error_code;
      return error_code;
    }
  } else if (arithmetic_run_ != nullptr) {
    error_code = arithmetic_run_(input0_data, input1_data, output_data, element_num, &quant_args_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Arithmetic run fail!ret: " << error_code;
      return error_code;
    }
  } else {
    MS_LOG(ERROR) << "arithmetic_run function is nullptr!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticInt8CPUKernel::Run() {
  auto param = reinterpret_cast<ArithmeticParameter *>(op_parameter_);
  if (param->broadcasting_) {
    auto input_data0 = reinterpret_cast<int8_t *>(in_tensors_[0]->MutableData());
    CHECK_NULL_RETURN(input_data0);
    auto input_data1 = reinterpret_cast<int8_t *>(in_tensors_[1]->MutableData());
    CHECK_NULL_RETURN(input_data1);
    MS_CHECK_GT(out_tensors_[0]->Size(), 0, RET_ERROR);
    tile_data0_ = reinterpret_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_[0]->Size()));
    if (tile_data0_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
    tile_data1_ = reinterpret_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_[0]->Size()));
    if (tile_data1_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      ms_context_->allocator->Free(tile_data0_);
      return RET_ERROR;
    }
    TileDimensionsInt8(input_data0, input_data1, tile_data0_, tile_data1_, param);
  }
  auto ret = ParallelLaunch(this->ms_context_, ArithmeticsInt8Launch, this, op_parameter_->thread_num_);
  if (param->broadcasting_) {
    ms_context_->allocator->Free(tile_data0_);
    ms_context_->allocator->Free(tile_data1_);
    tile_data0_ = nullptr;
    tile_data1_ = nullptr;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic launch function fail! ret: " << ret;
  }
  return ret;
}

kernel::InnerKernel *CpuArithmeticInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                    const lite::Context *ctx, const kernel::KernelKey &desc) {
  kernel::InnerKernel *kernel = nullptr;
  ArithmeticParameter *param = reinterpret_cast<ArithmeticParameter *>(parameter);
  if (desc.type == PrimitiveType_Eltwise && param->eltwise_mode_ == static_cast<int>(schema::EltwiseMode_SUM)) {
    kernel = new (std::nothrow)
      QuantizedAddCPUKernel(parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  } else if (desc.type == PrimitiveType_Eltwise && param->eltwise_mode_ == static_cast<int>(schema::EltwiseMode_PROD)) {
    kernel =
      new (std::nothrow) MulInt8CPUKernel(parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  } else {
    kernel = new (std::nothrow)
      ArithmeticInt8CPUKernel(parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create ArithmeticInt8CPUKernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Equal, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_NotEqual, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Less, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LessEqual, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Greater, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_GreaterEqual, CpuArithmeticInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Eltwise, CpuArithmeticInt8KernelCreator)
}  // namespace mindspore::kernel
