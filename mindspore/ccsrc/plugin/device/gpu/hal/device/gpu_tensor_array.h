/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_TENSOR_ARRAY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_TENSOR_ARRAY_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "runtime/device/tensor_array.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUTensorArray : public TensorArray {
 public:
  GPUTensorArray(const string &name, const TypePtr &dtype, const std::vector<size_t> &shapes)
      : TensorArray(name, dtype, shapes) {}
  ~GPUTensorArray() override = default;
  void ReleaseMemory(const DeviceMemPtr addr) override;
  void *CreateMemory(const size_t size) override;
  void ClearMemory(void *addr, const size_t size) override;
};
using GPUTensorArray = GPUTensorArray;
using GPUTensorArrayPtr = std::shared_ptr<GPUTensorArray>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_TENSOR_ARRAY_H_
