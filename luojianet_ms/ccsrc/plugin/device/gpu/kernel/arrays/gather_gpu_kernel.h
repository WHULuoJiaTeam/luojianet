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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename S>
class GatherFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherFwdGpuKernelMod() : axis_(0), is_null_input_(false) {}
  ~GatherFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *index_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    Gather(input_addr, index_addr, output_addr, dims_[0], dims_[1], dims_[2], dims_[3],
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    input_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    index_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name, "input") ||
                     CHECK_SHAPE_NULL(index_shapes_, kernel_name, "input_indices") ||
                     CHECK_SHAPE_NULL(output_shapes_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shapes_.size() != index_shapes_.size() || input_shapes_.size() != output_shapes_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output should be the equal to the "
                        << "dimension of index: " << index_shapes_.size()
                        << ", but got the dimension of input: " << input_shapes_.size()
                        << ", the dimension of output: " << output_shapes_.size();
    }
    int dims = SizeToInt(input_shapes_.size());
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dim"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }
    Reshape();
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {}
  void InitSizeLists() override {
    size_t size = GetSize(input_shapes_, true);
    input_size_list_.push_back(size);

    size = GetSize(index_shapes_, false);
    input_size_list_.push_back(size);

    size = GetSize(output_shapes_, true);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    size_t dim_before_axis = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dim_before_axis *= output_shapes_[i];
    }
    size_t dim_at_axis_input = input_shapes_[IntToSize(axis_)];
    size_t dim_at_axis_output = output_shapes_[IntToSize(axis_)];
    size_t dim_after_axis = 1;
    for (size_t i = IntToSize(axis_) + 1; i < output_shapes_.size(); i++) {
      dim_after_axis *= output_shapes_[i];
    }

    dims_[0] = dim_before_axis;
    dims_[1] = dim_at_axis_input;
    dims_[2] = dim_at_axis_output;
    dims_[3] = dim_after_axis;
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape, const bool flag = true) const {
    size_t result = flag ? sizeof(T) : sizeof(S);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> index_shapes_;
  std::vector<size_t> output_shapes_;

  size_t dims_[4] = {};
  int axis_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_GATHER_GPU_KERNEL_H_
