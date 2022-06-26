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

#ifndef LUOJIANET_MS_LITE_CONVOLUTION_FP32_H_
#define LUOJIANET_MS_LITE_CONVOLUTION_FP32_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/kernel.h"

namespace luojianet_ms::kernel {
class ConvolutionCPUFp32 : public InnerKernel {
 public:
  ConvolutionCPUFp32(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors,
                     std::vector<lite::Tensor *> out_tensors, const lite::Context *ctx);
  virtual ~ConvolutionCPUFp32();
  int Prepare() override;  // init, execute once

  int Run() override;
  int ReSize() override;

 private:
  KernelBase *kernel;
  TensorC *in[2];
  TensorC *out[1];
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_FP32_H_
