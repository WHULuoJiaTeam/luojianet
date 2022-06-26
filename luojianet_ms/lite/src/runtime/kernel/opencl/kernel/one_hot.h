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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ONE_HOT_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ONE_HOT_H_

#include <vector>
#include <string>
#include "src/inner_kernel.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/one_hot_fp32.h"

namespace luojianet_ms::kernel {
class OneHotOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~OneHotOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int InitWeights() override;
  int CheckSpecs() override;
  int SetConstArgs() override;
  int SetGlobalLocal() override;

 private:
  int depth_{0};
  float on_value_{1.0f};
  float off_value_{0.0f};
  int axis_{0};
  GpuTensorInfo in_shape_;
  GpuTensorInfo out_shape_;
  OneHotParameter *param_;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ONE_HOT_H_
