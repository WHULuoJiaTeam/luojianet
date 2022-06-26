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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_TRANSPOSE_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_TRANSPOSE_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/transpose.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace luojianet_ms::kernel {
enum class TransposeType { AXIS0312, AXIS0231, GENERAL };

class TransposeOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~TransposeOpenCLKernel() override = default;

  void BroadCastPerm();
  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  int SetConstArgs() override;
  int SetGlobalLocal() override;

 private:
  TransposeType type_{TransposeType::AXIS0312};
  GpuTensorInfo tensor_size_;
  int perm_4d_[4];
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_TRANSPOSE_H_
