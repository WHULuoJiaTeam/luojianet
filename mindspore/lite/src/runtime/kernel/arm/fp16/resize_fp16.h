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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_RESIZE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_RESIZE_FP16_H_

#include <vector>
#include <algorithm>
#include "src/runtime/kernel/arm/fp32/resize_fp32.h"
#include "nnacl/fp16/resize_fp16.h"

namespace mindspore::kernel {
class ResizeFp16CPUKernel : public ResizeCPUKernel {
 public:
  ResizeFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ResizeCPUKernel(parameter, inputs, outputs, ctx) {}

  ~ResizeFp16CPUKernel() = default;

 private:
  int ResizePrepare() override;
  int DataTypeLen() override;
  int RunImpl(int task_id) override;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_RESIZE_FP16_H_
