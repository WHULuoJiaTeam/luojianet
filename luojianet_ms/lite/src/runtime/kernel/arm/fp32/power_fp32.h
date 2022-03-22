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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_POWER_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_POWER_H_

#include <vector>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "nnacl/fp32/power_fp32.h"

namespace luojianet_ms::kernel {
class PowerCPUKernel : public InnerKernel {
 public:
  PowerCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(param, inputs, outputs, ctx),
        thread_count_(ctx->thread_num_),
        scale_(reinterpret_cast<PowerParameter *>(op_parameter_)->scale_),
        shift_(reinterpret_cast<PowerParameter *>(op_parameter_)->shift_) {}
  ~PowerCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id) const;

 private:
  int thread_count_;
  float scale_;
  float shift_;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_POWER_H_
