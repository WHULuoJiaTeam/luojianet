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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_POOLING_FP16_GRAD_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_POOLING_FP16_GRAD_H_

#include <vector>
#include "src/inner_kernel.h"

namespace luojianet_ms::kernel {
using luojianet_ms::schema::PadMode;
using luojianet_ms::schema::PoolMode;
using luojianet_ms::schema::QuantType;
using luojianet_ms::schema::RoundMode;

class PoolingGradCPUKernelFp16 : public InnerKernel {
 public:
  explicit PoolingGradCPUKernelFp16(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~PoolingGradCPUKernelFp16() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int Execute(int task_id);

 private:
  int thread_num_ = 1;
};

}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_POOLING_FP16_GRAD_H_
