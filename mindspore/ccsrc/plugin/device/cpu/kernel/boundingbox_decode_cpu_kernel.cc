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

#include "plugin/device/cpu/kernel/boundingbox_decode_cpu_kernel.h"
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void BoundingBoxDecodeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != INPUT_NUMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
  }

  const size_t coordinate_size = 4;
  if (common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<ValueTuple>() ||
      common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<ValueList>()) {
    means_ = common::AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, "means");
  } else if (common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<FloatImm>()) {
    float mean = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "means");
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)means_.emplace_back(mean);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'means' should be a tuple or a list, and dtype should be float, but got is not.";
  }

  if (common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<ValueTuple>() ||
      common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<ValueList>()) {
    stds_ = common::AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, "stds");
  } else if (common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<FloatImm>()) {
    float std = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "stds");
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)stds_.emplace_back(std);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'stds' should be a tuple or a list, and dtype should be float, but got is not.";
  }

  if (means_.size() < coordinate_size || stds_.size() < coordinate_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of input 'means' and 'stds' should be at least 4, "
                         "but got the length of 'means': "
                      << means_.size() << ", and the length of 'stds': " << stds_.size();
  }

  std::vector<int64_t> max_shape_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "max_shape");
  (void)std::transform(max_shape_me.begin(), max_shape_me.end(), std::back_inserter(max_shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  wh_ratio_clip_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "wh_ratio_clip");

  if (max_shape_.size() < MIN_MAX_SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of 'max_shape' should be at least 2, but got: " << max_shape_.size();
  }

  InitTaskFunc(kernel_node);
}

template <typename T>
bool BoundingBoxDecodeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  auto anchor_box = reinterpret_cast<T *>(inputs[0]->addr);
  auto deltas = reinterpret_cast<T *>(inputs[1]->addr);
  auto bboxes = reinterpret_cast<T *>(outputs[0]->addr);

  T ms1 = static_cast<T>(max_shape_[0]);
  T ms2 = static_cast<T>(max_shape_[1]);

  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dtype of inputs 'anchor_box' and 'deltas' should be the same, "
                     "but got the memory size of 'anchor_box': "
                  << inputs[0]->size << " and 'deltas': " << inputs[1]->size;
    return false;
  }

  const size_t coordinate = 4;
  const size_t block_size = inputs[0]->size / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the memory size of input 'anchor_box' must be a multiple of 4, "
                     "but got the memory size of 'anchor_box': "
                  << inputs[0]->size;
    return false;
  }

  size_t elem_num = block_size / coordinate;
  auto task = [this, &deltas, &anchor_box, &bboxes, ms1, ms2](size_t start, size_t end) {
    constexpr size_t X_INDEX = 0;
    constexpr size_t Y_INDEX = 1;
    constexpr size_t W_INDEX = 2;
    constexpr size_t H_INDEX = 3;
    const T ZERO = static_cast<T>(0);
    const T HALF = static_cast<T>(0.5);
    const T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      const size_t left_x = i * 4;
      const size_t left_y = i * 4 + 1;
      const size_t right_x = i * 4 + 2;
      const size_t right_y = i * 4 + 3;

      T dx = deltas[left_x] * static_cast<T>(stds_[X_INDEX]) + static_cast<T>(means_[X_INDEX]);
      T dy = deltas[left_y] * static_cast<T>(stds_[Y_INDEX]) + static_cast<T>(means_[Y_INDEX]);
      T dw = deltas[right_x] * static_cast<T>(stds_[W_INDEX]) + static_cast<T>(means_[W_INDEX]);
      T dh = deltas[right_y] * static_cast<T>(stds_[H_INDEX]) + static_cast<T>(means_[H_INDEX]);

      T max_ratio = static_cast<T>(abs(log(wh_ratio_clip_)));

      dw = dw > max_ratio ? max_ratio : (dw < (-max_ratio) ? (-max_ratio) : dw);
      dh = dh > max_ratio ? max_ratio : (dh < (-max_ratio) ? (-max_ratio) : dh);

      T px = (anchor_box[left_x] + anchor_box[right_x]) * HALF;
      T py = (anchor_box[left_y] + anchor_box[right_y]) * HALF;
      T pw = anchor_box[right_x] - anchor_box[left_x] + ONE;
      T ph = anchor_box[right_y] - anchor_box[left_y] + ONE;

      T gx = px + pw * dx;
      T gy = py + ph * dy;
      T gw = pw * exp(dw);
      T gh = ph * exp(dh);

      T x1 = gx - gw * HALF + HALF;
      T y1 = gy - gh * HALF + HALF;
      T x2 = gx + gw * HALF - HALF;
      T y2 = gy + gh * HALF - HALF;

      x1 = x1 > ms2 ? ms2 : (x1 < ZERO ? ZERO : x1);
      y1 = y1 > ms1 ? ms1 : (y1 < ZERO ? ZERO : y1);
      x2 = x2 > ms2 ? ms2 : (x2 < ZERO ? ZERO : x2);
      y2 = y2 > ms1 ? ms1 : (y2 < ZERO ? ZERO : y2);

      bboxes[left_x] = x1;
      bboxes[left_y] = y1;
      bboxes[right_x] = x2;
      bboxes[right_y] = y2;
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, BoundingBoxDecodeCpuKernelMod::BoundingBoxDecodeFunc>>
  BoundingBoxDecodeCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BoundingBoxDecodeCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BoundingBoxDecodeCpuKernelMod::LaunchKernel<float16>}};

void BoundingBoxDecodeCpuKernelMod::InitTaskFunc(const CNodePtr &kernel_node) {
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "BoundingBoxDecode does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

std::vector<KernelAttr> BoundingBoxDecodeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BoundingBoxDecodeFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BoundingBoxDecode, BoundingBoxDecodeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
