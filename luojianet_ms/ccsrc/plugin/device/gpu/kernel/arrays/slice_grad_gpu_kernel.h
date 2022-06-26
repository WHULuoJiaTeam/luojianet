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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class SliceGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SliceGradGpuKernelMod()
      : is_strided_slice_(false),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        kernel_name_("SliceGrad") {}
  ~SliceGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    FillDeviceArray(outputs[0]->size / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalSlice4DGrad(begin_[0], begin_[1], begin_[2], begin_[3], size_[0], size_[1], size_[2], size_[3], input_shape_[0],
                   input_shape_[1], input_shape_[2], input_shape_[3], dy, dx,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckParam(kernel_node);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto data_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    if (kernel_name == "StridedSliceGrad") {
      is_strided_slice_ = true;
      std::vector<int64_t> shapex = GetAttr<std::vector<int64_t>>(kernel_node, "shapex");
      for (auto x : shapex) {
        input_shape_.push_back(static_cast<size_t>(x));
      }
      for (auto i = input_shape_.size(); i < 4; i++) {
        (void)input_shape_.insert(input_shape_.begin(), 1);
      }
      strides_ = GetAttr<std::vector<int64_t>>(kernel_node, "strides");
      for (auto i = strides_.size(); i < 4; i++) {
        (void)strides_.insert(strides_.begin(), 1);
      }
      size_ = GetAttr<std::vector<int64_t>>(kernel_node, "end");
    } else {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
      is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      ShapeNdTo4d(input_shape, &input_shape_);
      size_ = GetAttr<std::vector<int64_t>>(kernel_node, "size");
    }
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    ShapeNdTo4d(dy_shape, &dy_shape_);
    begin_ = GetAttr<std::vector<int64_t>>(kernel_node, "begin");
    CalcBeginAndSize(data_format);
    input_size_ = input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(T);
    output_size_ = sizeof(T);
    for (auto x : dy_shape_) {
      output_size_ = output_size_ * x;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(output_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
  }
  void CalcBeginAndSize(const std::string data_format) {
    for (auto i = begin_.size(); i < 4; i++) {
      (void)begin_.insert(begin_.begin(), 0);
    }
    for (auto i = size_.size(); i < 4; i++) {
      (void)size_.insert(size_.begin(), 1);
    }
    if (data_format == "NHWC") {
      std::swap(begin_[1], begin_[3]);
      std::swap(begin_[1], begin_[2]);
      std::swap(size_[1], size_[3]);
      std::swap(size_[1], size_[2]);
    }
    for (size_t i = 0; i < begin_.size(); i++) {
      if (begin_[i] < 0 && i < input_shape_.size()) {
        begin_[i] = begin_[i] + input_shape_[i];
      }
    }
    for (size_t i = 0; i < size_.size(); i++) {
      if (size_[i] < 0 && i < input_shape_.size()) {
        size_[i] = (size_[i] + input_shape_[i]) > 0 ? (size_[i] + input_shape_[i]) : 0;
      }
    }
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() > 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than 4, but got "
                        << input_shape.size();
    }
  }

  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::vector<int64_t> strides_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> dy_shape_;

  bool is_strided_slice_;
  bool is_null_input_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::string kernel_name_;
};  // namespace kernel
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
