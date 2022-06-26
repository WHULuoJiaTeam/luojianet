/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SCATTER_ND_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SCATTER_ND_BASE_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/base/scatter_nd_base.h"

namespace mindspore::kernel {
class ScatterNDCPUKernel : public InnerKernel {
 public:
  explicit ScatterNDCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ScatterNDParameter *>(parameter);
  }
  ~ScatterNDCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int ScatterND(int task_id);

 private:
  ScatterNDParameter *param_ = nullptr;
  std::vector<int> output_unit_offsets_;
  std::vector<int> out_strides_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SCATTER_ND_BASE_H_
