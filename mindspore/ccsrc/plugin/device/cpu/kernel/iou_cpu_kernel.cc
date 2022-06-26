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

#include "plugin/device/cpu/kernel/iou_cpu_kernel.h"

#include <string>
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIOUInputsNum = 2;
constexpr size_t kIOUOutputsNum = 1;
constexpr size_t kBoxCoordinateLen = 4;
}  // namespace

void IOUCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto anchor_boxes_shape = AnfAlgo::GetInputDeviceShape(kernel_node, ANCHOR_BOXES);
  constexpr size_t BOX_SHAPE_SIZE = 2;
  constexpr size_t BOX_SIZE_INDEX = 0;
  constexpr size_t BOX_COORDINATE_INDEX = 1;

  if (anchor_boxes_shape.size() != BOX_SHAPE_SIZE || anchor_boxes_shape[BOX_COORDINATE_INDEX] != kBoxCoordinateLen) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'anchor_boxes' should be [N, 4], but got: " << Vector2Str(anchor_boxes_shape);
  }
  anchor_boxes_size_ = anchor_boxes_shape[BOX_SIZE_INDEX];
  auto gt_boxes_shape = AnfAlgo::GetInputDeviceShape(kernel_node, GT_BOXES);
  if (gt_boxes_shape.size() != BOX_SHAPE_SIZE || gt_boxes_shape[BOX_COORDINATE_INDEX] != kBoxCoordinateLen) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'gt_boxes' should be [N, 4], but got: " << Vector2Str(gt_boxes_shape);
  }
  gt_boxes_size_ = gt_boxes_shape[BOX_SIZE_INDEX];
  iou_size_ = anchor_boxes_size_ * gt_boxes_size_;
  std::string iou_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  if (iou_mode != "iou" && iou_mode != "iof") {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'mode' should be 'iou' or 'iof', but got: " << iou_mode;
  }
  if (iou_mode == "iof") {
    mode_ = IOF_MODE;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "IOU does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool IOUCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIOUInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIOUOutputsNum, kernel_name_);
  auto anchor_boxes = reinterpret_cast<T *>(inputs[ANCHOR_BOXES]->addr);
  auto gt_boxes = reinterpret_cast<T *>(inputs[GT_BOXES]->addr);
  auto iou_score = reinterpret_cast<T *>(outputs[IOU_VALUE]->addr);

  // multithreading
  auto task = [&anchor_boxes, &gt_boxes, &iou_score, this](size_t start, size_t end) {
    const T ZERO = T(0);
    const T ONE = T(1);
    const T EPS = T(1e-10);
    constexpr size_t Y0_SHIFT = 1;
    constexpr size_t X1_SHIFT = 2;
    constexpr size_t Y1_SHIFT = 3;
    for (size_t i = start; i < end; i++) {
      size_t idx1 = i % anchor_boxes_size_ * kBoxCoordinateLen;
      size_t idx2 = i / anchor_boxes_size_ * kBoxCoordinateLen;
      T I_x0 = std::max(anchor_boxes[idx1], gt_boxes[idx2]);
      T I_y0 = std::max(anchor_boxes[idx1 + Y0_SHIFT], gt_boxes[idx2 + Y0_SHIFT]);
      T I_x1 = std::min(anchor_boxes[idx1 + X1_SHIFT], gt_boxes[idx2 + X1_SHIFT]);
      T I_y1 = std::min(anchor_boxes[idx1 + Y1_SHIFT], gt_boxes[idx2 + Y1_SHIFT]);
      T overlaps_w = std::max(ZERO, (I_x1 - I_x0 + ONE));
      T overlaps_h = std::max(ZERO, (I_y1 - I_y0 + ONE));
      T overlaps = overlaps_w * overlaps_h;
      T area1 = (anchor_boxes[idx1 + X1_SHIFT] - anchor_boxes[idx1] + ONE) *
                (anchor_boxes[idx1 + Y1_SHIFT] - anchor_boxes[idx1 + Y0_SHIFT] + ONE);
      T area2 = (gt_boxes[idx2 + X1_SHIFT] - gt_boxes[idx2] + ONE) *
                (gt_boxes[idx2 + Y1_SHIFT] - gt_boxes[idx2 + Y0_SHIFT] + ONE);
      if (mode_ == IOU_MODE) {
        iou_score[i] = overlaps / (area1 + area2 - overlaps + EPS);
      } else {
        iou_score[i] = overlaps / (area2 + EPS);
      }
    }
  };
  ParallelLaunchAutoSearch(task, iou_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, IOUCpuKernelMod::IOUFunc>> IOUCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &IOUCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &IOUCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> IOUCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IOUFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IOU, IOUCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
