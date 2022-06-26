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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DEPTH_TO_SPACE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DEPTH_TO_SPACE_FP16_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/depth_to_space_fp32.h"

namespace mindspore::kernel {
class DepthToSpaceFp16CPUKernel : public DepthToSpaceCPUKernel {
 public:
  DepthToSpaceFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DepthToSpaceCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DepthToSpaceFp16CPUKernel() = default;

  int Prepare() override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DEPTH_TO_SPACE_FP16_H_
