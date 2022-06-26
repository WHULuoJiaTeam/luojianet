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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_SEQUENCE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_SEQUENCE_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/reverse_sequence_fp32.h"

namespace mindspore::kernel {
class ReverseSequenceCPUKernel : public InnerKernel {
 public:
  ReverseSequenceCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~ReverseSequenceCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  void ConvertAxisToPositive(const std::vector<int> shape, int *axis) const;
  int CalcCountPreAxis(const std::vector<int> shape, int axis) const;
  int CalcCountAfterAxis(const std::vector<int> shape, int axis) const;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_SEQUENCE_H_
