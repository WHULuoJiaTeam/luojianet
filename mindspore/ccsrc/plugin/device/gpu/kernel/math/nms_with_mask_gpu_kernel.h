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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <iostream>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nms_with_mask_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class NMSWithMaskFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  NMSWithMaskFwdGpuKernelMod()
      : num_input_(0),
        iou_value_(0.5),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        ceil_power_2(0) {}
  ~NMSWithMaskFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *data_buff = GetDeviceAddress<T>(workspace, 0);
    int *index_buff = GetDeviceAddress<int>(workspace, 1);
    bool *row_mask = GetDeviceAddress<bool>(workspace, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int *sel_idx = GetDeviceAddress<int>(outputs, 1);
    bool *sel_boxes = GetDeviceAddress<bool>(outputs, 2);

    CalSort(num_input_, input, output, index_buff, data_buff, box_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalPreprocess(num_input_, sel_idx, sel_boxes, input, output, index_buff, box_size_, row_mask,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    CalNms(num_input_, iou_value_, output, sel_boxes, box_size_, row_mask, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    iou_value_ = GetAttr<float>(kernel_node, "iou_threshold");

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }

    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 3, but got " << output_num;
    }

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    num_input_ = input_shape[0];  // Get N value in [N,5] data
    ceil_power_2 = NmsRoundUpPower2(num_input_);

    input_size_ = num_input_ * sizeof(T) * box_size_;  // 5 values per bbox
    output_size_ = (input_size_) + (num_input_ * sizeof(int)) + (num_input_ * sizeof(bool));

    workspace_size_ = ceil_power_2 * (sizeof(T) + sizeof(int));   // sorting buffers
    workspace_size_ += (num_input_ * num_input_ * sizeof(bool));  // Row mask - NMS

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    // N sized input/output data
    input_size_list_.push_back(num_input_ * sizeof(T) * box_size_);
    output_size_list_.push_back(num_input_ * sizeof(T) * box_size_);
    output_size_list_.push_back(num_input_ * sizeof(int));
    output_size_list_.push_back(num_input_ * sizeof(bool));

    // N sized workspace arrs
    workspace_size_list_.push_back(ceil_power_2 * sizeof(T));                // data buff
    workspace_size_list_.push_back(ceil_power_2 * sizeof(int));              // index buff
    workspace_size_list_.push_back(num_input_ * num_input_ * sizeof(bool));  // mask list
  }

 private:
  int num_input_;
  float iou_value_;
  bool is_null_input_;
  static const int box_size_ = 5;  // pre_defined box width
  // default values
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  size_t ceil_power_2;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NMS_WITH_MASK_GPU_KERNEL_H_
