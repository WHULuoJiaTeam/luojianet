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

#include "plugin/device/cpu/kernel/resize_nearest_neighbor_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborInputsNum = 1;
constexpr size_t kResizeNearestNeighborOutputNum = 1;
constexpr size_t kResizeNearestNeighborInputsShapeSize = 4;
constexpr size_t kResizeNearestNeighborAttrSize = 2;
}  // namespace

void ResizeNearestNeighborCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<int64_t> output_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
  align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (input_shape.size() != kResizeNearestNeighborInputsShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_x' should be "
                      << kResizeNearestNeighborInputsShapeSize << ", but got " << input_shape.size();
  }

  if (output_size.size() != kResizeNearestNeighborAttrSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'size' should be "
                      << kResizeNearestNeighborAttrSize << ", but got " << output_size.size();
  }

  batch_size_ = input_shape[0];
  channel_ = input_shape[1];
  in_height_ = input_shape[2];
  in_width_ = input_shape[3];
  out_height_ = LongToSize(output_size[0]);
  out_width_ = LongToSize(output_size[1]);
  height_scale_ = Scaling(in_height_, out_height_, align_corners_);
  width_scale_ = Scaling(in_width_, out_width_, align_corners_);
  output_size_ = batch_size_ * channel_ * out_height_ * out_width_;
}

bool ResizeNearestNeighborCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborOutputNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be float16, float32, float64, int32, or int64, but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T>
void ResizeNearestNeighborCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &outputs) {
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  if (out_height_ == in_height_ && out_width_ == in_width_) {
    for (size_t i = 0; i < output_size_; ++i) {
      output_addr[i] = input_addr[i];
    }
  }

  for (size_t i = 0; i < output_size_; ++i) {
    size_t pos0 = i / (channel_ * out_height_ * out_width_) % batch_size_;
    size_t pos1 = i / (out_height_ * out_width_) % channel_;
    size_t pos2 = i / (out_width_) % out_height_;
    size_t pos3 = i % out_width_;
    const size_t in_y = std::min((align_corners_) ? static_cast<size_t>(roundf(pos2 * height_scale_))
                                                  : static_cast<size_t>(floorf(pos2 * height_scale_)),
                                 in_height_ - 1);
    const size_t in_x = std::min((align_corners_) ? static_cast<size_t>(roundf(pos3 * width_scale_))
                                                  : static_cast<size_t>(floorf(pos3 * width_scale_)),
                                 in_width_ - 1);
    size_t input_pos =
      pos0 * channel_ * in_height_ * in_width_ + pos1 * in_height_ * in_width_ + in_y * in_width_ + in_x;
    output_addr[i] = input_addr[input_pos];
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighbor, ResizeNearestNeighborCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
