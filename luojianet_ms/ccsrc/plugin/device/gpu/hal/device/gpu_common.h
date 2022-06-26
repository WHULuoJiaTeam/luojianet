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

#ifndef LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_GPU_GPU_COMMON_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_GPU_GPU_COMMON_H_

#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <sstream>
#include "utils/log_adapter.h"
#include "utils/trace_base.h"
#include "include/curand.h"

namespace luojianet_ms {
namespace device {
namespace gpu {
#define CHECK_OP_RET_WITH_EXCEPT(expression, message)                                 \
  {                                                                                   \
    bool success = (expression);                                                      \
    if (!success) {                                                                   \
      MS_LOG(EXCEPTION) << "Op Error: " << message << " | Error Number: " << success; \
    }                                                                                 \
  }

#define CHECK_OP_RET_WITH_EXCEPT_TRANCE(node, expression, message)                                  \
  {                                                                                                 \
    bool success = (expression);                                                                    \
    if (!success) {                                                                                 \
      MS_LOG(EXCEPTION) << "Op Error: " << message << " | " << trace::DumpSourceLines(node.lock()); \
    }                                                                                               \
  }

#define CHECK_OP_RET_WITH_ERROR(expression, message)                              \
  {                                                                               \
    bool success = (expression);                                                  \
    if (!success) {                                                               \
      MS_LOG(ERROR) << "Op Error: " << message << " | Error Number: " << success; \
    }                                                                             \
  }

#define CHECK_RET_WITH_RETURN_ERROR(expression, message) \
  {                                                      \
    bool success = (expression);                         \
    if (!success) {                                      \
      MS_LOG(ERROR) << message;                          \
      return false;                                      \
    }                                                    \
  }

#define CHECK_CUDA_RET_WITH_ERROR(node, expression, message)                                                           \
  {                                                                                                                    \
    cudaError_t status = (expression);                                                                                 \
    if (status != cudaSuccess) {                                                                                       \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " << cudaGetErrorString(status) \
                    << trace::DumpSourceLines(node.lock());                                                            \
    }                                                                                                                  \
  }

#define CHECK_CUDA_RET_WITH_ERROR_NOTRACE(expression, message)                           \
  {                                                                                      \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
    }                                                                                    \
  }

#define CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(expression, message)                    \
  {                                                                                      \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
      return false;                                                                      \
    }                                                                                    \
  }

#define CHECK_CUDA_RET_WITH_EXCEPT(node, expression, message)                                 \
  {                                                                                           \
    cudaError_t status = (expression);                                                        \
    if (status != cudaSuccess) {                                                              \
      MS_LOG(EXCEPTION) << "CUDA Error: " << message << " | Error Number: " << status << " "  \
                        << cudaGetErrorString(status) << trace::DumpSourceLines(node.lock()); \
    }                                                                                         \
  }

#define CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(expression, message)                              \
  {                                                                                          \
    cudaError_t status = (expression);                                                       \
    if (status != cudaSuccess) {                                                             \
      MS_LOG(EXCEPTION) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                        << cudaGetErrorString(status);                                       \
    }                                                                                        \
  }

#define CHECK_CUDNN_RET_WITH_EXCEPT(node, expression, message)                                 \
  {                                                                                            \
    cudnnStatus_t status = (expression);                                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                                      \
      MS_LOG(EXCEPTION) << "cuDNN Error: " << message << " | Error Number: " << status << " "  \
                        << cudnnGetErrorString(status) << trace::DumpSourceLines(node.lock()); \
    }                                                                                          \
  }

#define CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(expression, message)                              \
  {                                                                                           \
    cudnnStatus_t status = (expression);                                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                                                     \
      MS_LOG(EXCEPTION) << "cuDNN Error: " << message << " | Error Number: " << status << " " \
                        << cudnnGetErrorString(status);                                       \
    }                                                                                         \
  }

#define CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(expression, message)                           \
  {                                                                                       \
    cudnnStatus_t status = (expression);                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                                 \
      MS_LOG(ERROR) << "cuDNN Error: " << message << " | Error Number: " << status << " " \
                    << cudnnGetErrorString(status);                                       \
    }                                                                                     \
  }

#define CHECK_CUDNN_RET_WITH_ERROR(node, expression, message)                              \
  {                                                                                        \
    cudnnStatus_t status = (expression);                                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                                                  \
      MS_LOG(ERROR) << "cuDNN Error: " << message << " | Error Number: " << status << " "  \
                    << cudnnGetErrorString(status) << trace::DumpSourceLines(node.lock()); \
    }                                                                                      \
  }

#define CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(expression, message)                              \
  {                                                                                            \
    cublasStatus_t status = (expression);                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
      MS_LOG(EXCEPTION) << "cuBLAS Error: " << message << " | Error Number: " << status << " " \
                        << luojianet_ms::device::gpu::cuBlasGetErrorString(status);               \
    }                                                                                          \
  }

#define CHECK_CUBLAS_RET_WITH_EXCEPT(node, expression, message)                                \
  {                                                                                            \
    cublasStatus_t status = (expression);                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
      MS_LOG(EXCEPTION) << "cuBLAS Error: " << message << " | Error Number: " << status << " " \
                        << luojianet_ms::device::gpu::cuBlasGetErrorString(status)                \
                        << trace::DumpSourceLines(node.lock());                                \
    }                                                                                          \
  }

#define CHECK_CUBLAS_RET_WITH_ERROR(expression, message)                                   \
  {                                                                                        \
    cublasStatus_t status = (expression);                                                  \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                 \
      MS_LOG(ERROR) << "cuBLAS Error: " << message << " | Error Number: " << status << " " \
                    << luojianet_ms::device::gpu::cuBlasGetErrorString(status);               \
    }                                                                                      \
  }

#define CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(expression, message)                        \
  {                                                                                        \
    cusolverStatus_t status = (expression);                                                \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                               \
      MS_LOG(EXCEPTION) << "cusolver Error: " << message << " | Error Number: " << status; \
    }                                                                                      \
  }

#define CHECK_CUSOLVER_RET_WITH_EXCEPT(node, expression, message)                         \
  {                                                                                       \
    cusolverStatus_t status = (expression);                                               \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                              \
      MS_LOG(EXCEPTION) << "cusolver Error: " << message << " | Error Number: " << status \
                        << trace::DumpSourceLines(node.lock());                           \
      ;                                                                                   \
    }                                                                                     \
  }

#define CHECK_CUSOLVER_RET_WITH_ERROR(expression, message)                             \
  {                                                                                    \
    cusolverStatus_t status = (expression);                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                           \
      MS_LOG(ERROR) << "cusolver Error: " << message << " | Error Number: " << status; \
    }                                                                                  \
  }

#define CHECK_NCCL_RET_WITH_EXCEPT(node, expression, message)                         \
  {                                                                                   \
    int result = (expression);                                                        \
    if (result != ncclSuccess) {                                                      \
      MS_LOG(EXCEPTION) << "NCCL Error: " << message << " | Error Number: " << result \
                        << trace::DumpSourceLines(node.lock());                       \
    }                                                                                 \
  }

#define VARIABLE_NOT_USED(var) \
  { (void)(var); }

inline bool CheckNullInput(const std::vector<size_t> &input_shape) {
  // If input_shape.size() == 0, it means a scalar input; If input_shape.size() != 0 and input_shape contains 0,
  // it means a null input. Just return a null output.
  if (input_shape.size() != 0) {
    if (std::any_of(input_shape.begin(), input_shape.end(), [](size_t i) { return i == 0; })) {
      return true;
    }
  }
  return false;
}
#define CHECK_NULL_INPUT(input_shape) luojianet_ms::device::gpu::CheckNullInput(input_shape)

template <typename T>
inline std::string ConvertVectorToString(const std::vector<T> &value) {
  std::stringstream ss;
  ss << "(";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it == value.begin()) {
      ss << *it;
    } else {
      ss << ", " << *it;
    }
  }
  ss << ")";
  return ss.str();
}

#define CONVERT_VECTOR_TO_STRING(value) luojianet_ms::device::gpu::ConvertVectorToString(value)

inline bool CheckShapeNull(const std::vector<size_t> &shape, std::string kernel_name, std::string param_name) {
  if (CHECK_NULL_INPUT(shape)) {
    MS_LOG(WARNING) << "For '" << kernel_name << "', the shape of " << param_name << " cannot contain zero, but got "
                    << CONVERT_VECTOR_TO_STRING(shape);
    return true;
  }
  return false;
}

#define CHECK_SHAPE_NULL(shape, kernel_name, param_name) \
  luojianet_ms::device::gpu::CheckShapeNull(shape, kernel_name, param_name)

inline const char *CurandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_VERSION_MISMATCH:
      return "Header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
      return "Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "Length requested is not a multiple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "Preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "Initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "Architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "Internal library error.";
    default:
      return "Unknown the curandStatus.";
  }
}

inline const char *cuBlasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS: The operation completed successfully.";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED: The cuBLAS library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuBLAS library. This is usually caused "
             "by a cudaMalloc() failure. ";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE: An unsupported value or parameter was passed to the function (a negative "
             "vector size, for example).";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture; "
             "usually caused by compute capability lower than 5.0.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR: An access to GPU memory space failed, which is usually caused by a failure "
             "to bind a texture.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED: The GPU program failed to execute. This is often caused by a launch "
             "failure of the kernel on the GPU, which can be caused by multiple reasons.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR: An internal cuBLAS operation failed. This error is usually caused by a "
             "cudaMemcpyAsync() failure. ";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED: The functionality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR: The functionality requested requires some license and an error was detected "
             "when trying to check the current licensing. This error can happen if the license is not present or is "
             "expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly. ";
    default:
      return "Unknown cublasStatus.";
  }
}

#define CHECK_CURAND_RET_WITH_EXCEPT(expression, message)                                           \
  {                                                                                                 \
    curandStatus_t status = (expression);                                                           \
    if (status != CURAND_STATUS_SUCCESS) {                                                          \
      MS_LOG(EXCEPTION) << "CUDA curand Error: " << message << " | curandStatus: " << status << " " \
                        << luojianet_ms::device::gpu::CurandGetErrorString(status);                    \
    }                                                                                               \
  }
}  // namespace gpu
}  // namespace device
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_GPU_GPU_COMMON_H_
