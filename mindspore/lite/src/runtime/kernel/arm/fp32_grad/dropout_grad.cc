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

#include <random>
#include "src/runtime/kernel/arm/fp32_grad/dropout_grad.h"
#include "nnacl/fp32_grad/dropout_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32_grad/dropout_parameter.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DropoutGrad;

namespace mindspore::kernel {
constexpr static int kInputIdx = 0;
constexpr static int kMaskIdx = 1;
constexpr static int kOutputIdx = 0;

int DropoutGradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_2D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIdx));
  CHECK_NULL_RETURN(in_tensors_.at(kMaskIdx));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIdx));
  CHECK_NULL_RETURN(op_parameter_);
  auto param = reinterpret_cast<DropoutParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "Dropout op_parameter_ nullptr";
    return RET_NULL_PTR;
  }

  if ((param->ratio_ > 1.0f) || (param->ratio_ < 0.0f)) {
    MS_LOG(ERROR) << "unsupported ratio value - Dropout ratio should be between zero to one";
    return RET_ERROR;
  }

  if (param->ratio_ >= 1.0f) {
    scale_ = 1.0f;
  } else {
    scale_ = 1. / (1. - param->ratio_);
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DropoutGradCPUKernel::ReSize() { return RET_OK; }

int DropoutGradCPUKernel::DoExecute(int task_id) {
  auto yt_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIdx)->MutableData());
  auto mask_ptr = reinterpret_cast<float *>(in_tensors_.at(kMaskIdx)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIdx)->MutableData());
  CHECK_NULL_RETURN(yt_ptr);
  CHECK_NULL_RETURN(mask_ptr);
  CHECK_NULL_RETURN(output_ptr);
  auto length = in_tensors_.at(kInputIdx)->ElementsNum();
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count > 0) {
    int start = stride * task_id;
    DropoutGrad(&(yt_ptr[start]), &(mask_ptr[start]), &(output_ptr[start]), count, scale_);
  }
  return RET_OK;
}

int RunDropoutGrad(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto dropout = reinterpret_cast<DropoutGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(dropout);
  auto error_code = dropout->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Dropout Grad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DropoutGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, RunDropoutGrad, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Dropout Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::InnerKernel *CpuDropoutGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "DropoutGrad opParameter nullptr.";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_DropoutGrad) {
    MS_LOG(ERROR) << "DropoutGrad desc type should be " << schema::PrimitiveType_DropoutGrad << " got " << desc.type;
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) DropoutGradCPUKernel(opParameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "DropoutGrad new kernel failed.";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DropoutGrad, CpuDropoutGradFp32KernelCreator)
}  // namespace mindspore::kernel
