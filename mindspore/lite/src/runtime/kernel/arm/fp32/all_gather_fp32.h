/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ALL_GATHER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ALL_GATHER_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/all_gather_parameter.h"

namespace mindspore::kernel {
class AllGatherCPUKernel : public InnerKernel {
 public:
  AllGatherCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<AllGatherParameter *>(parameter);
  }
  ~AllGatherCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  AllGatherParameter *param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ALL_GATHER_H_
