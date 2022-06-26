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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore::kernel {
class ActivationGradCPUKernel : public InnerKernel {
 public:
  explicit ActivationGradCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(param, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    param_act_grad_ = reinterpret_cast<ActivationParameter *>(param);
  }
  ~ActivationGradCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  ActivationParameter *param_act_grad_;
  int thread_count_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_
