/**
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SOFTMAX_BASE_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SOFTMAX_BASE_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/softmax_parameter.h"

namespace luojianet_ms::kernel {
class SoftmaxBaseCPUKernel : public InnerKernel {
 public:
  SoftmaxBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    softmax_param_ = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  }
  ~SoftmaxBaseCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override { return 0; }

 protected:
  int thread_count_;
  SoftmaxParameter *softmax_param_;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_SOFTMAX_BASE_H_
