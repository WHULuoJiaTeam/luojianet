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

#include "src/delegate/tensorrt/tensorrt_allocator.h"
#include <cuda_runtime.h>
#include <mutex>
#include "src/common/log_adapter.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
void *TensorRTAllocator::MallocDeviceMem(const mindspore::MSTensor &host_tensor, size_t size) {
  if (host_tensor == NULL) {
    return nullptr;
  }
  return MallocDeviceMem(host_tensor.Name(), size, ConvertDataType(host_tensor.DataType()));
}

void *TensorRTAllocator::MallocDeviceMem(const std::string &name, size_t size, nvinfer1::DataType data_type) {
  if (cuda_tensor_map_.find(name) != cuda_tensor_map_.end() && size <= cuda_tensor_map_[name].size) {
    MS_LOG(DEBUG) << "tensor :" << name << " has already in cuda Allocator pool.";
    return cuda_tensor_map_[name].data;
  }
  void *device_ptr = nullptr;
  auto cuda_ret = cudaMalloc(&device_ptr, size);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda Malloc failed for size:" << size;
    return nullptr;
  }
  MS_LOG(INFO) << "cudaMalloc size: " << size << " for " << name;
  if (cuda_tensor_map_[name].data != nullptr) {
    cuda_ret = cudaFree(cuda_tensor_map_[name].data);
    if (cuda_ret != cudaSuccess && cuda_ret != cudaErrorCudartUnloading) {
      MS_LOG(ERROR) << "free old cuda device_ptr failed for " << cudaGetErrorName(cuda_ret);
      cuda_ret = cudaFree(device_ptr);
      if (cuda_ret != cudaSuccess) {
        MS_LOG(ERROR) << "free new cuda device_ptr failed for " << cudaGetErrorName(cuda_ret);
        return nullptr;
      }
      return nullptr;
    }
  }
  cuda_tensor_map_[name].data = device_ptr;
  cuda_tensor_map_[name].is_valid_mem = false;
  cuda_tensor_map_[name].size = size;
  return device_ptr;
}

void TensorRTAllocator::MarkMemValid(const std::string &name, bool isValid) {
  cuda_tensor_map_[name].is_valid_mem = isValid;
  return;
}

bool TensorRTAllocator::GetMemIsValid(const std::string &name) {
  if (cuda_tensor_map_.find(name) == cuda_tensor_map_.end()) {
    MS_LOG(WARNING) << "tensor :" << name << " not in cuda Allocator pool.";
    return false;
  }
  return cuda_tensor_map_[name].is_valid_mem;
}

void *TensorRTAllocator::GetDevicePtr(const std::string &tensor_name) {
  if (tensor_name.empty()) {
    return nullptr;
  }
  if (cuda_tensor_map_.find(tensor_name) == cuda_tensor_map_.end()) {
    return nullptr;
  }
  return this->cuda_tensor_map_.find(tensor_name)->second.data;
}

int TensorRTAllocator::SyncMemInHostAndDevice(mindspore::MSTensor host_tensor, const std::string &device_tensor_name,
                                              bool is_host2device, bool sync) {
  if (host_tensor == NULL) {
    MS_LOG(ERROR) << "host tensor is null.";
    return RET_ERROR;
  }
  return SyncMemInHostAndDevice(host_tensor.MutableData(), device_tensor_name, host_tensor.DataSize(), is_host2device,
                                sync);
}

int TensorRTAllocator::SyncMemInHostAndDevice(void *host_data, const std::string &device_tensor_name, size_t data_size,
                                              bool is_host2device, bool sync) {
  if (host_data == nullptr || cuda_tensor_map_.find(device_tensor_name) == cuda_tensor_map_.end()) {
    MS_LOG(ERROR) << " host or device ptr is null.";
    return RET_ERROR;
  }
  CudaTensorParam &current_cuda_tensor = cuda_tensor_map_.find(device_tensor_name)->second;
  // is memcpy from device to host, the host mem is valid, change tag for mem pool.
  current_cuda_tensor.is_valid_mem = is_host2device ? current_cuda_tensor.is_valid_mem : true;
  if (is_host2device && current_cuda_tensor.is_valid_mem) {
    MS_LOG(DEBUG) << "no need memcpy for: " << device_tensor_name;
    return RET_OK;
  }
  auto device_ptr = current_cuda_tensor.data;
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "device_ptr is null for " << device_tensor_name;
    return RET_ERROR;
  }

  void *src_ptr = is_host2device ? host_data : device_ptr;
  void *dst_ptr = is_host2device ? device_ptr : host_data;
  cudaMemcpyKind kind = is_host2device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  auto cuda_ret = cudaMemcpy(dst_ptr, src_ptr, data_size, kind);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "copy mem failed,ret " << cudaGetErrorName(cuda_ret);
    return RET_ERROR;
  }
  MS_LOG(INFO) << "cuda memcpy success for " << device_tensor_name;
  return RET_OK;
}

int TensorRTAllocator::ClearDeviceMem() {
  for (auto &iter : cuda_tensor_map_) {
    auto cuda_ret = cudaFree(iter.second.data);
    if (cuda_ret != cudaSuccess && cuda_ret != cudaErrorCudartUnloading) {
      MS_LOG(WARNING) << "free cuda failed for " << cudaGetErrorName(cuda_ret);
    }
    iter.second.data = nullptr;
    iter.second.is_valid_mem = false;
  }
  return RET_OK;
}
std::map<std::string, CudaTensorParam> TensorRTAllocator::GetAllDevicePtr() { return this->cuda_tensor_map_; }
}  // namespace mindspore::lite
