/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/environ/environ_gpu_set.h"
#include "kernel/environ_manager.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"

namespace mindspore {
namespace kernel {
bool EnvironSetGpuKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  if (!EnvironMgr::GetInstance().CheckEnvInput(kernel_node)) {
    MS_LOG(ERROR) << "The input checks invalid, kernel: " << kernel_node->fullname_with_scope();
    return false;
  }

  // Check the output handle.
  auto handle_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  auto handle_shapes = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(ERROR) << "The output handle checks invalid, kernel: " << kernel_node->fullname_with_scope();
    return false;
  }

  value_type_attr_ = TypeId(common::AnfAlgo::GetNodeAttr<int>(kernel_node, kEnvValueTypeAttr));
  MS_LOG(INFO) << "The EnvironSet kernel " << kernel_node->fullname_with_scope() << " value type: " << value_type_attr_;
  handle_size_ = sizeof(int64_t);
  key_size_ = sizeof(int64_t);

  auto value_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 2);
  auto value_shapes = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
  value_size_ = GetTypeByte(TypeIdToType(value_type));
  for (auto &i : value_shapes) {
    value_size_ *= i;
  }

  InitSizeLists();
  return true;
}

void EnvironSetGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(handle_size_);
  input_size_list_.push_back(key_size_);
  input_size_list_.push_back(value_size_);
  output_size_list_.push_back(handle_size_);
}

bool EnvironSetGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input_handle = GetDeviceAddress<int64_t>(inputs, 0);
  auto input_key = GetDeviceAddress<int64_t>(inputs, 1);
  auto input_value = GetDeviceAddress<void>(inputs, 2);
  auto output_handle = GetDeviceAddress<int64_t>(outputs, 0);

  // Get host handle and host key.
  int64_t host_handle = 0;
  int64_t host_key = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&host_handle, input_handle, handle_size_, cudaMemcpyDeviceToHost,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Get handle failed.");
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&host_key, input_key, key_size_, cudaMemcpyDeviceToHost,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Get key failed.");
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Sync stream failed.");

  // Alloc the value address, and free in the step end.
  auto value_ptr = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(value_size_);
  MS_EXCEPTION_IF_NULL(value_ptr);
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(value_ptr, input_value, value_size_, cudaMemcpyDeviceToDevice,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Copy value failed.");

  // Set env member.
  const auto &env = EnvironMgr::GetInstance().Get(host_handle);
  if (env == nullptr) {
    MS_LOG(EXCEPTION) << "Get the env failed, handle: " << host_handle << ", key: " << host_key;
  }
  auto env_value = std::make_shared<EnvironValue>(value_ptr, value_size_, value_type_attr_, kGPUDevice);
  env->Set(host_key, env_value);

  // Copy output handle.
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(output_handle, input_handle, handle_size_, cudaMemcpyDeviceToDevice,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Copy output handle failed.");

  return true;
}
}  // namespace kernel
}  // namespace mindspore
