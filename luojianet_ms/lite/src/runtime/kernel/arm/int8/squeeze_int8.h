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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SQUEEZE_INT8_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SQUEEZE_INT8_H_

#include <vector>
#include "include/context.h"
#include "include/errorcode.h"
#include "src/inner_kernel.h"
#include "nnacl/int8/squeeze_int8.h"
#include "nnacl/squeeze_parameter.h"

using luojianet_ms::lite::InnerContext;
namespace luojianet_ms::kernel {
class SqueezeInt8CPUKernel : public InnerKernel {
 public:
  SqueezeInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~SqueezeInt8CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  void DoExecute(int tId);

 private:
  SqueezeQuantArg *quant_squeeze_param_{nullptr};
};

int SqueezeInt8Run(void *cdata, int task_id, float, float);
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_SQUEEZE_INT8_H_
