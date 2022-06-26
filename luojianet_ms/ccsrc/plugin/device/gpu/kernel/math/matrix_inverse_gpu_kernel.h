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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <type_traits>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class MatrixInverseGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixInverseGpuKernelMod()
      : input_size_(0), adjoint_(false), is_null_input_(false), handle_(nullptr), batch_size_(1), size_(1) {}
  ~MatrixInverseGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    auto compute_input_addr = GetDeviceAddress<T>(workspace, 0);
    auto lu_batch_addr = GetDeviceAddress<T *>(workspace, 1);
    auto inv_batch_addr = GetDeviceAddress<T *>(workspace, 2);
    auto pivo_addr = GetDeviceAddress<int>(workspace, 3);
    auto info_addr = GetDeviceAddress<int>(workspace, 4);

    int len = SizeToInt(size_);
    int batchsize = SizeToInt(batch_size_);
    for (size_t i = 0; i < batch_size_; i++) {
      lu_addr_[i] = compute_input_addr + i * len * len;
      inv_addr_[i] = output_addr + i * len * len;
    }
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(compute_input_addr, input_addr, input_size_, cudaMemcpyDeviceToDevice,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(lu_batch_addr, lu_addr_.data(), sizeof(T *) * batch_size_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(inv_batch_addr, inv_addr_.data(), sizeof(T *) * batch_size_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    if (std::is_same<T, float>::value) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasSgetrfBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr), len,
                                                       pivo_addr, info_addr, batchsize),
                                   "cublas trsm batched Fail");
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasSgetriBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr), len, pivo_addr,
                            reinterpret_cast<float **>(inv_batch_addr), len, info_addr, batchsize),
        "cublas trsm batched Fail");
    } else if (std::is_same<T, double>::value) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasDgetrfBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr), len,
                                                       pivo_addr, info_addr, batchsize),
                                   "cublas trsm batched Fail");
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasDgetriBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr), len, pivo_addr,
                            reinterpret_cast<double **>(inv_batch_addr), len, info_addr, batchsize),
        "cublas trsm batched Fail");
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type entered must be float or double.";
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
                        << input_shape.size();
    }
    size_t last_index = input_shape.size() - 1;
    if (input_shape[last_index] != input_shape[last_index - 1]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the last two dimensions of the input matrix should be equal, "
                        << "but got one: " << input_shape[last_index] << ", another: " << input_shape[last_index - 1];
    }
    size_ = input_shape[last_index];
    for (size_t i = 0; i < last_index - 1; i++) {
      batch_size_ *= input_shape[i];
    }

    input_size_ = sizeof(T);
    for (auto dim : input_shape) {
      input_size_ *= dim;
    }
    adjoint_ = GetAttr<bool>(kernel_node, "adjoint");
    lu_addr_.resize(batch_size_);
    inv_addr_.resize(batch_size_);
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.emplace_back(input_size_);
    output_size_list_.emplace_back(input_size_);
    workspace_size_list_.emplace_back(input_size_);
    size_t lu_size = batch_size_ * sizeof(T *);
    workspace_size_list_.emplace_back(lu_size);
    size_t inv_size = batch_size_ * sizeof(T *);
    workspace_size_list_.emplace_back(inv_size);
    size_t pivo_size = batch_size_ * size_ * sizeof(int);
    workspace_size_list_.emplace_back(pivo_size);
    size_t info_size = batch_size_ * sizeof(int);
    workspace_size_list_.emplace_back(info_size);
  }

 private:
  size_t input_size_;
  bool adjoint_;
  bool is_null_input_;
  cublasHandle_t handle_;
  size_t batch_size_;
  size_t size_;
  std::vector<T *> lu_addr_;
  std::vector<T *> inv_addr_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
