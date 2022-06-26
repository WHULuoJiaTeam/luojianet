/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 4;
constexpr size_t kInputIndexForN = 0;
constexpr size_t kInputIndexForC = 1;
constexpr size_t kInputIndexForH = 2;
constexpr size_t kInputIndexForW = 3;
constexpr size_t kOutputIndexForH = 2;
constexpr size_t kOutputIndexForW = 3;

template <typename T, typename S>
class MaxPoolWithArgmaxFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  MaxPoolWithArgmaxFwdGpuKernelMod()
      : n_(0),
        c_(0),
        input_height_(0),
        input_width_(0),
        window_height_(0),
        window_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        stride_height_(0),
        stride_width_(0),
        output_height_(0),
        output_width_(0),
        is_null_input_(false),
        input_size_(0),
        output_size_(0) {}
  ~MaxPoolWithArgmaxFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    S *index_addr = GetDeviceAddress<S>(outputs, 1);
    CalMaxPoolWithArgmax(input_addr, n_, c_, input_height_, input_width_, window_height_, window_width_, stride_height_,
                         stride_width_, pad_top_, pad_left_, output_height_, output_width_, output_addr, index_addr,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 2, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = sizeof(T);
    for (auto x : input_shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(T);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    if (input_shape.size() < kInputDimLowerLimit || output_shape.size() < kOutputDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input and output cannot be less than 4, but "
                        << "got the dimension of input: " << input_shape.size()
                        << ", the dimension of output: " << output_shape.size();
    }
    n_ = SizeToInt(input_shape[kInputIndexForN]);
    c_ = SizeToInt(input_shape[kInputIndexForC]);
    input_height_ = SizeToInt(input_shape[kInputIndexForH]);
    input_width_ = SizeToInt(input_shape[kInputIndexForW]);
    output_height_ = SizeToInt(output_shape[kOutputIndexForH]);
    output_width_ = SizeToInt(output_shape[kOutputIndexForW]);
    std::vector<int> window;
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the length of 'kernel_size' cannot be less than 3, but got "
                        << window.size();
    }
    window_height_ = window[1];
    window_width_ = window[2];
    std::vector<int> stride;
    std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride.size() < 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the length of 'strides' cannot be less than 3, but got "
                        << stride.size();
    }
    stride_height_ = stride[1];
    stride_width_ = stride[2];
    pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
    pad_top_ = 0;
    pad_left_ = 0;
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      SetPad();
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_ / sizeof(T) * sizeof(S));
  }

 private:
  void SetPad() {
    MS_EXCEPTION_IF_ZERO("stride height", stride_height_);
    MS_EXCEPTION_IF_ZERO("stride width", stride_width_);

    int tmp_height = (input_height_ / stride_height_) * stride_height_ == input_height_
                       ? (input_height_ / stride_height_)
                       : (input_height_ / stride_height_) + 1;
    pad_height_ = std::max<int>(0, (tmp_height - 1) * stride_height_ + window_height_ - input_height_);

    int tmp_width = (input_width_ / stride_width_) * stride_width_ == input_width_ ? (input_width_ / stride_width_)
                                                                                   : (input_width_ / stride_width_) + 1;
    pad_width_ = std::max<int>(0, (tmp_width - 1) * stride_width_ + window_width_ - input_width_);
    pad_top_ = pad_height_ / 2;
    pad_left_ = pad_width_ / 2;
  }

  std::string pad_mode_;

  int n_;
  int c_;
  int input_height_;
  int input_width_;
  int window_height_;
  int window_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int stride_height_;
  int stride_width_;
  int output_height_;
  int output_width_;
  bool is_null_input_;

  size_t input_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GPU_KERNEL_H_
