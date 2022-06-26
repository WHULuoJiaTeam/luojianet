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

#ifndef LUOJIANET_MS_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H
#define LUOJIANET_MS_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/iou_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class IOUGpuKernelMod : public NativeGpuKernelMod {
 public:
  IOUGpuKernelMod() : gt_boxes_size_(0), anchor_boxes_size_(0), iou_size_(0), mode_(0), is_null_input_(false) {}
  ~IOUGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *gt_boxes_addr = GetDeviceAddress<T>(inputs, 0);
    T *anchor_boxes_addr = GetDeviceAddress<T>(inputs, 1);
    T *iou_addr = GetDeviceAddress<T>(outputs, 0);

    const size_t coordinate = 4;
    const size_t block_size_0 = inputs[0]->size / sizeof(T);
    const size_t block_size_1 = inputs[1]->size / sizeof(T);
    if ((block_size_0 % coordinate) != 0 || (block_size_1 % coordinate) != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
      return false;
    }

    const size_t input_len_0 = block_size_0 / coordinate;
    const size_t input_len_1 = block_size_1 / coordinate;
    IOU(input_len_0 * input_len_1, gt_boxes_addr, anchor_boxes_addr, iou_addr, mode_, input_len_0,
        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    gt_boxes_size_ = sizeof(T);
    anchor_boxes_size_ = sizeof(T);
    iou_size_ = sizeof(T);

    auto gt_boxes_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto anchor_boxes_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto iou_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(gt_boxes_shape, kernel_name_, "anchor_boxes") ||
                     CHECK_SHAPE_NULL(anchor_boxes_shape, kernel_name_, "gt_boxes") ||
                     CHECK_SHAPE_NULL(iou_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < gt_boxes_shape.size(); i++) {
      gt_boxes_size_ *= gt_boxes_shape[i];
    }

    for (size_t i = 0; i < anchor_boxes_shape.size(); i++) {
      anchor_boxes_size_ *= anchor_boxes_shape[i];
    }

    for (size_t i = 0; i < iou_shape.size(); i++) {
      iou_size_ *= iou_shape[i];
    }

    InitSizeLists();

    std::string mode = GetAttr<std::string>(kernel_node, "mode");

    if (mode == "iou") {
      mode_ = 0;
    } else if (mode == "iof") {
      mode_ = 1;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', mode only support 'iou' or 'iof'.";
    }

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(gt_boxes_size_);
    input_size_list_.push_back(anchor_boxes_size_);
    output_size_list_.push_back(iou_size_);
  }

 private:
  size_t gt_boxes_size_;
  size_t anchor_boxes_size_;
  size_t iou_size_;
  size_t mode_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_KERNEL_GPU_OTHER_IOU_GPU_KERNEL_H
