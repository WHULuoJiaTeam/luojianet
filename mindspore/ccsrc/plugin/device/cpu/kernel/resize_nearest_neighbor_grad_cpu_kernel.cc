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

#include "plugin/device/cpu/kernel/resize_nearest_neighbor_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeNearestNeighborGradInputsNum = 1;
constexpr size_t kResizeNearestNeighborGradOutputNum = 1;
constexpr size_t kResizeNearestNeighborGradInputsShapeSize = 4;
constexpr size_t kResizeNearestNeighborGradOutputsShapeSize = 4;
}  // namespace

void ResizeNearestNeighborGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> output_size = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  align_corners_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
  dtype_ = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);

  if (input_shape.size() != kResizeNearestNeighborGradInputsShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be "
                      << kResizeNearestNeighborGradInputsShapeSize << ", but got " << input_shape.size();
  }

  if (output_size.size() != kResizeNearestNeighborGradOutputsShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be "
                      << kResizeNearestNeighborGradOutputsShapeSize << ", but got " << output_size.size();
  }

  batch_size_ = input_shape[0];
  channel_ = input_shape[1];
  in_height_ = input_shape[2];
  in_width_ = input_shape[3];
  out_height_ = output_size[2];
  out_width_ = output_size[3];
  height_scale_ = Scaling(out_height_, in_height_, align_corners_);
  width_scale_ = Scaling(out_width_, in_width_, align_corners_);
}

bool ResizeNearestNeighborGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeNearestNeighborGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeNearestNeighborGradOutputNum, kernel_name_);
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
void ResizeNearestNeighborGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &outputs) {
  const auto *dloss_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed. Error no: " << ret;
  }

  size_t in_hw_size = in_width_ * in_height_;
  size_t out_hw_size = out_width_ * out_height_;
  for (size_t b = 0; b < batch_size_; ++b) {
    for (size_t c = 0; c < channel_; ++c) {
      for (size_t h = 0; h < in_height_; ++h) {
        const size_t out_y = std::min((align_corners_) ? static_cast<size_t>(roundf(h * height_scale_))
                                                       : static_cast<size_t>(floorf(h * height_scale_)),
                                      out_height_ - 1);
        for (size_t w = 0; w < in_width_; ++w) {
          const size_t out_x = std::min((align_corners_) ? static_cast<size_t>(roundf(w * width_scale_))
                                                         : static_cast<size_t>(floorf(w * width_scale_)),
                                        out_width_ - 1);
          output_addr[out_y * out_width_ + out_x] += dloss_addr[h * in_width_ + w];
        }
      }
      output_addr += out_hw_size;
      dloss_addr += in_hw_size;
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeNearestNeighborGrad, ResizeNearestNeighborGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
