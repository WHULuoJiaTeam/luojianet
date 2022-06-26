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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_

#include <vector>
#include "src/inner_kernel.h"

namespace luojianet_ms::kernel {
class ConvolutionGradFilterCPUKernel : public InnerKernel {
 public:
  explicit ConvolutionGradFilterCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)

      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionGradFilterCPUKernel() override {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  size_t ws_size_ = 0;
  bool do_img2col_ = true;
  bool do_dw_ = false;
  std::mutex lock_;
  size_t mat_alloc_ = 0;
#ifdef ENABLE_ARM32
  const int chunk_ = C4NUM * 2;
#else
  const int chunk_ = C12NUM * 2;
#endif
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_
