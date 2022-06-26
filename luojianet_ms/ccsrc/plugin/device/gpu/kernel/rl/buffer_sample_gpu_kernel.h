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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_BUFFER_SAMPLE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_BUFFER_SAMPLE_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
class BufferSampleKernelMod : public NativeGpuKernelMod {
 public:
  BufferSampleKernelMod();
  ~BufferSampleKernelMod();

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
  void ReleaseResource() override;

 protected:
  void InitSizeLists() override;

 private:
  size_t element_nums_;
  int64_t capacity_;
  size_t batch_size_;
  int64_t seed_;
  bool states_init_;
  bool unique_;
  std::mt19937 generator_;
  curandState *devStates_;
  std::vector<size_t> exp_element_list;
};

MS_REG_GPU_KERNEL(BufferSample, BufferSampleKernelMod)
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RL_BUFFER_SAMPLE_GPU_KERNEL_H_
