/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_

#include <cudnn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cusolverDn.h>

#define CUDA_LIB_EXPORT __attribute__((visibility("default")))
#define CUDA_KERNEL_ASSERT(cond)                                                       \
  if (!(cond)) {                                                                       \
    __assert_fail(#cond, __FILE__, static_cast<unsigned int>(__LINE__), __FUNCTION__); \
  }
namespace mindspore {
namespace device {
namespace gpu {
class CudaCommon {
 public:
  inline int threads_num() const { return threads_per_block_; }
  inline int threads_num(int size) const { return std::min(size, threads_per_block_); }
  inline int major_sm() const { return major_sm_; }
  inline float cuda_cap() const { return static_cast<float>(major_sm_ * 10 + minor_sm_) / 10.0; }
  inline int blocks_num(const int total_threads) const {
    return std::min(((total_threads - 1) / threads_per_block_) + 1, max_blocks_);
  }
  size_t share_memory_size() const { return max_share_memory_; }
  void set_check_sm(const bool &flag) { check_sm_ = flag; }
  bool check_sm() const { return check_sm_; }

  static CudaCommon &GetInstance();

 private:
  CudaCommon();
  ~CudaCommon() = default;
  CudaCommon(const CudaCommon &) = delete;
  CudaCommon &operator=(const CudaCommon &) = delete;

  int max_blocks_;
  int threads_per_block_;
  int major_sm_;
  int minor_sm_;
  size_t max_share_memory_;
  bool check_sm_{true};
};
#define GET_BLOCKS(total_threads) mindspore::device::gpu::CudaCommon::GetInstance().blocks_num(total_threads)
#define GET_THREADS mindspore::device::gpu::CudaCommon::GetInstance().threads_num()
#define GET_THREADS_MAXSIZE(size) mindspore::device::gpu::CudaCommon::GetInstance().threads_num(size)
#define GET_MAJOR_SM mindspore::device::gpu::CudaCommon::GetInstance().major_sm()
#define GET_CUDA_CAP mindspore::device::gpu::CudaCommon::GetInstance().cuda_cap()
#define SHARED_MEM_PER_BLOCK mindspore::device::gpu::CudaCommon::GetInstance().share_memory_size()
#define MINIUM_SM 6
#define RECOMMEND_SM 7
#define SUPPORTED_CAP 5.3
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CUDA_COMMON_H_
