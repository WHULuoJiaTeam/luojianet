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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHER_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHER_INT8_H_

#include <vector>
#include "nnacl/gather_parameter.h"
#include "nnacl/int8/quantize.h"
#include "src/inner_kernel.h"

namespace mindspore::kernel {
class GatherInt8CPUKernel : public InnerKernel {
 public:
  GatherInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}
  ~GatherInt8CPUKernel() {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGather(int task_id);

 private:
  int thread_count_;
  int axis_{0};
  GatherQuantArg param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHER_INT8_H_
