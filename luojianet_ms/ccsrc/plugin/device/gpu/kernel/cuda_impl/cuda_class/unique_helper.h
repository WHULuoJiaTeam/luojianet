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

#ifndef LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_HELPER_H_
#define LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_HELPER_H_
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unique_impl.cuh"

namespace luojianet_ms {
namespace cukernel {
constexpr size_t INPUT_NUM = 1;
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t WORK_NUM = 0;
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t CROPS_SHAPE_0 = 2;
constexpr size_t CROPS_SHAPE_1 = 2;

template <typename T, typename S>
class UniqueHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit UniqueHelperGpuKernel(std::string &kernel_name) : GpuKernelHelperBase(kernel_name) {}
  virtual ~UniqueHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<size_t>> &input_shapes,
                 const std::vector<std::vector<size_t>> &output_shapes) override {
    int flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (flag != 0) {
      return flag;
    }
    num_elements_ = input_size_list_[0] / sizeof(T);
    size_t workspace_size = num_elements_ * sizeof(S);
    work_size_list_.emplace_back(workspace_size);
    work_size_list_.emplace_back(workspace_size);
    output_size_list_.emplace_back(input_size_list_[0]);
    output_size_list_.emplace_back(num_elements_ * sizeof(S));
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    T *t_input_ptr = nullptr;
    S *s_input_index = nullptr;
    S *s_sorted_index = nullptr;
    T *t_output_ptr = nullptr;
    S *s_output_index = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &t_input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(work_ptrs, 0, kernel_name_, &s_input_index);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(work_ptrs, 1, kernel_name_, &s_sorted_index);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &t_output_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(output_ptrs, 1, kernel_name_, &s_output_index);
    if (flag != 0) {
      return flag;
    }

    post_output_size_ = CalUnique(t_input_ptr, num_elements_, s_input_index, s_sorted_index, t_output_ptr,
                                  s_output_index, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void ResetResource() override {
    num_elements_ = 1;
    post_output_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

  int GetOutSize() { return post_output_size_; }

 private:
  int num_elements_;
  int post_output_size_;
};
}  // namespace cukernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNIQUE_HELPER_H_
