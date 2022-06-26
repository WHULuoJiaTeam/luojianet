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

#ifndef LUOJIANET_MS_DATASET_INIT_KERNEL_H
#define LUOJIANET_MS_DATASET_INIT_KERNEL_H

#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
class DatasetInitKernelMod : public NativeGpuKernelMod {
 public:
  DatasetInitKernelMod();
  ~DatasetInitKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 protected:
  void InitSizeLists() override;

 private:
  std::string queue_name_;
  std::vector<size_t> shapes_;
  size_t total_bytes_;

  // The capacity of buffer Q.
  size_t buffer_q_capacity_{2};
};

MS_REG_GPU_KERNEL(InitDataSetQueue, DatasetInitKernelMod)
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_QUEUE_CPU_KERNEL_H
