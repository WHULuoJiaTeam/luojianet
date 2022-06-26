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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_UNSQUEEZE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_UNSQUEEZE_H_

#include <vector>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "nnacl/int8/unsqueeze_int8.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class Unsqueezeint8CPUKernel : public InnerKernel {
 public:
  Unsqueezeint8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    param_ = reinterpret_cast<UnSqueezeParameter *>(op_parameter_);
    param_->thread_count_ = op_parameter_->thread_num_;
  }
  ~Unsqueezeint8CPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoUnsqueeze(int task_id);

 private:
  UnSqueezeParameter *param_{nullptr};
  int thread_sz_count_{0};
  int thread_sz_stride_{0};
  int data_size_{0};
  float *in_ptr_{nullptr};
  float *out_ptr_{nullptr};
  int thread_count_{0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_UNSQUEEZE_H_
