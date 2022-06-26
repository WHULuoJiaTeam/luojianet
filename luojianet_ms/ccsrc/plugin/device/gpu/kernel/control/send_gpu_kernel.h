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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_SEND_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_SEND_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
class SendGpuKernelMod : public NativeGpuKernelMod {
 public:
  SendGpuKernelMod() {}
  ~SendGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *) override {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaEventRecord(record_event_, record_stream_),
                               "Recording cuda event failed.");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    record_stream_ = reinterpret_cast<cudaStream_t>(GetAttr<uintptr_t>(kernel_node, "record_event_stream"));
    record_event_ = reinterpret_cast<cudaEvent_t>(GetAttr<uintptr_t>(kernel_node, "record_event"));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    return;
  }

 private:
  cudaStream_t record_stream_{nullptr};
  cudaEvent_t record_event_{nullptr};
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CONTROL_SEND_GPU_KERNEL_H_
