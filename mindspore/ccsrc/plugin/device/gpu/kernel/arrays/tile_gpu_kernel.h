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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tile_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class TileGpuKernelMod : public NativeGpuKernelMod {
 public:
  TileGpuKernelMod() { ResetResource(); }
  ~TileGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    size_t *input_shape_ptr = GetDeviceAddress<size_t>(workspace, 0);
    size_t *output_shape_ptr = GetDeviceAddress<size_t>(workspace, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_ptr, &input_shape_[0], input_shape_.size() * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(output_shape_ptr, &output_shape_[0], output_shape_.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync output_shape_ failed");
    CalTile(output_size_, input_size_, shape_size_, input_shape_ptr, output_shape_ptr, input, output,
            reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape_, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (output_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output cannot be less than 1, but got "
                        << output_shape_.size();
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input_size_ *= input_shape_[i];
    }

    output_size_ = 1;
    if (output_shape_.size() > TILE_MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of output cannot be greater than "
                        << TILE_MAX_DIMENSION << ", but got " << output_shape_.size();
    }
    shape_size_ = output_shape_.size();
    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }
    std::vector<int64_t> multiples = GetAttr<std::vector<int64_t>>(kernel_node, "multiples");
    int64_t filling_value = static_cast<int64_t>(multiples.size()) - static_cast<int64_t>(input_shape_.size());
    // input_shape_.size() == output_shape_.size() == shape_size_
    (void)input_shape_.insert(input_shape_.begin(), LongToSize(filling_value), 1);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    output_size_ = 1;
    shape_size_ = 1;
    is_null_input_ = false;
    input_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    workspace_size_list_.push_back(input_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(output_shape_.size() * sizeof(size_t));
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t shape_size_;
  bool is_null_input_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TILE_GPU_KERNEL_H_
