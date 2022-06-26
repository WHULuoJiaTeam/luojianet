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

#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

namespace luojianet_ms {
namespace device {
namespace gpu {
// ReleaseMemory() used in Free() in TensorArray.
void GPUTensorArray::ReleaseMemory(const DeviceMemPtr addr) {
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(addr);
}

void GPUTensorArray::ClearMemory(void *addr, const size_t size) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(addr, 0, size), "failed to set cuda memory with zeros.");
}

void *GPUTensorArray::CreateMemory(const size_t size) {
  return device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(size);
}
}  // namespace gpu
}  // namespace device
}  // namespace luojianet_ms
