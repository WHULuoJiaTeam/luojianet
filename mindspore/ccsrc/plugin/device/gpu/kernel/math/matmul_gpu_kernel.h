/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_impl.cuh"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kDimLowerLimit = 2;
constexpr size_t kDimOffset2 = 2;

template <typename T, typename S>
class MatMulGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatMulGpuKernelMod() { ResetResource(); }
  ~MatMulGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    VARIABLE_NOT_USED(workspace);
    if (is_null_input_) {
      return true;
    }
    auto input1_addr = GetDeviceAddress<T>(inputs, 0);
    auto input2_addr = GetDeviceAddress<T>(inputs, 1);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);

    // The alpha & beta should be float if the inputs are half, else should keep the same type with inputs.
    // The kernel registration is responsible for types consistency.
    S alpha = static_cast<S>(1.0f);
    S beta = static_cast<S>(0.0f);
    cudaDataType_t compute_type = (dtype_a_ == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F;

    const int lda = (transpose_x1_ == CUBLAS_OP_T) ? SizeToInt(m_) : SizeToInt(k_);
    const int ldb = (transpose_x2_ == CUBLAS_OP_T) ? SizeToInt(k_) : SizeToInt(n_);
    const int ldc = n_;

    auto stride_a = SizeToInt(m_ * k_);
    auto stride_b = SizeToInt(k_ * n_);
    auto stride_c = SizeToInt(m_ * n_);

    try {
      if (is_fused_matmul_biasadd_) {
        auto input3_addr = GetDeviceAddress<T>(inputs, 2);
        Fill(m_, n_, input3_addr, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
        beta = static_cast<S>(1.0f);
      }

      // Use cublasGemmEx to get high performance when batch_ is 1
      if (batch_ == 1) {
        CHECK_CUBLAS_RET_WITH_EXCEPT(
          kernel_node_,
          cublasGemmEx(handle_, transpose_x2_, transpose_x1_, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_), &alpha,
                       input2_addr, dtype_b_, ldb, input1_addr, dtype_a_, lda, &beta, output_addr, dtype_c_, ldc,
                       compute_type, algo_),
          "cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes.");
      } else {
        CHECK_CUBLAS_RET_WITH_EXCEPT(
          kernel_node_,
          cublasGemmStridedBatchedEx(handle_, transpose_x2_, transpose_x1_, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_),
                                     &alpha, input2_addr, dtype_b_, ldb, stride_b, input1_addr, dtype_a_, lda, stride_a,
                                     &beta, output_addr, dtype_c_, ldc, stride_c, batch_, compute_type, algo_),
          "cublasGemmStridedBatchedEx failed. Possible reasons: the GPU is occupied by other processes.");
      }
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', encountered an exception: " << e.what() << " when invoke "
                        << "cublas " << (batch_ == 1 ? "cublasGemmEx" : "cublasGemmStridedBatchedEx");
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    dtype_a_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    dtype_b_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 1)));
    dtype_c_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetOutputDeviceDataType(kernel_node, 0)));
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (dtype_a_ != dtype_b_ || dtype_a_ != dtype_c_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input and output types are not the same in " << node_name;
    }
    if (dtype_a_ == CUDA_R_16F && dtype_b_ == CUDA_R_16F && dtype_c_ == CUDA_R_16F) {
      MS_LOG(INFO) << "input and output type is float16, allow to use Tensor Core operations if possible";
      algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    auto input1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input1_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    auto dims = output_shape.size();
    if (dims < kDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 2, but got "
                        << dims;
    }

    m_ = output_shape[dims - kDimOffset2];
    n_ = output_shape[dims - 1];
    batch_ = 1;
    for (size_t i = 0; i < dims - kDimOffset2; i++) {
      batch_ *= output_shape[i];
    }

    bool transpose = GetAttr<bool>(kernel_node, "transpose_x1");
    transpose_x1_ = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (transpose && input1_shape.size() > (dims - kDimOffset2)) {
      k_ = input1_shape[dims - kDimOffset2];
    } else if (!transpose && input1_shape.size() > (dims - 1)) {
      k_ = input1_shape[dims - 1];
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', init k_ via input1_shape failed.";
    }

    transpose = GetAttr<bool>(kernel_node, "transpose_x2");
    transpose_x2_ = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    const std::string &kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kFusedMatMulBiasAddName) {
      is_fused_matmul_biasadd_ = true;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    batch_ = 0;
    m_ = 0;
    n_ = 0;
    k_ = 0;
    is_null_input_ = false;
    transpose_x1_ = CUBLAS_OP_N;
    transpose_x2_ = CUBLAS_OP_N;
    handle_ = nullptr;
    dtype_a_ = CUDA_R_32F;
    dtype_b_ = CUDA_R_32F;
    dtype_c_ = CUDA_R_32F;
    algo_ = CUBLAS_GEMM_DEFAULT;
    is_fused_matmul_biasadd_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t unit_size = sizeof(T);

    size_t input_size = batch_ * m_ * k_ * unit_size;
    input_size_list_.push_back(input_size);

    input_size = batch_ * n_ * k_ * unit_size;
    input_size_list_.push_back(input_size);

    size_t output_size = batch_ * m_ * n_ * unit_size;
    output_size_list_.push_back(output_size);
  }

 private:
  size_t batch_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool is_null_input_;

  cublasOperation_t transpose_x1_;
  cublasOperation_t transpose_x2_;
  cublasHandle_t handle_;
  cudaDataType_t dtype_a_;
  cudaDataType_t dtype_b_;
  cudaDataType_t dtype_c_;
  cublasGemmAlgo_t algo_;

  bool is_fused_matmul_biasadd_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_
