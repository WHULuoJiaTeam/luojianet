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

#include <cstring>
#include <vector>
#include "src/runtime/kernel/arm/fp16/softmax_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/softmax_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {
int SoftmaxFp16CPUKernel::Prepare() {
  auto ret = SoftmaxBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SoftmaxFp16CPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  auto in_shape = in_tensors_.front()->shape();
  out_plane_size_ = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size_ *= in_shape[i];
  }
  in_plane_size_ = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size_ *= in_shape[i];
  }
  if (in_plane_size_ > 1) {
    if (sum_data_ != nullptr) {
      free(sum_data_);
    }
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, out_plane_size_ * in_plane_size_ * sizeof(float16_t));
    sum_data_ = reinterpret_cast<float16_t *>(malloc(out_plane_size_ * in_plane_size_ * sizeof(float16_t)));
    if (sum_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc data for softmax fail!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SoftmaxFp16CPUKernel::DoSoftmaxLastAxis(int task_id) {
  int unit = UP_DIV(out_plane_size_, op_parameter_->thread_num_);
  if (INT_MUL_OVERFLOW(task_id, unit)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, out_plane_size_);
  int channel = softmax_param_->input_shape_[softmax_param_->axis_];
  if (INT_MUL_OVERFLOW(begin, channel)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int offset = begin * channel;
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(kInputIndex)->data());
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(kOutputIndex)->data());
  SoftmaxLastAxisFp16(input_ptr + offset, output_ptr + offset, end - begin, channel);
  return RET_OK;
}

int SoftmaxLastAxisFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<SoftmaxFp16CPUKernel *>(cdata);
  auto ret = kernel->DoSoftmaxLastAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSoftmaxLastAxisFp16 error task_id: " << task_id << ", ret: " << ret;
  }
  return ret;
}

int SoftmaxFp16CPUKernel::Run() {
  if (in_plane_size_ == 1) {
    auto ret = ParallelLaunch(this->ms_context_, SoftmaxLastAxisFp16Run, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SoftmaxFp16CPUKernel ParallelLaunch failed, ret: " << ret;
    }
    return ret;
  } else {
    auto input_tensor = in_tensors_.at(0);
    MS_ASSERT(input_tensor != nullptr);
    auto output_tensor = out_tensors_.at(0);
    MS_ASSERT(output_tensor != nullptr);
    input_fp16_ = reinterpret_cast<float16_t *>(input_tensor->data());
    MS_ASSERT(input_fp16_ != nullptr);
    output_fp16_ = reinterpret_cast<float16_t *>(output_tensor->data());
    MS_ASSERT(output_fp16_ != nullptr);
    MS_ASSERT(sum_data_ != nullptr);
    SoftmaxFp16(input_fp16_, output_fp16_, sum_data_, softmax_param_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Softmax, LiteKernelCreator<SoftmaxFp16CPUKernel>)
}  // namespace mindspore::kernel
