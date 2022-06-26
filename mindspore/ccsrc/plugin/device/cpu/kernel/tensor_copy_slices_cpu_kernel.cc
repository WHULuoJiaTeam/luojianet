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

#include "plugin/device/cpu/kernel/tensor_copy_slices_cpu_kernel.h"
#include <functional>
#include <unordered_map>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTensorCopySlicesInputsNum = 2;
constexpr size_t kTensorCopySlicesOutputsNum = 1;
}  // namespace

void TensorCopySlicesCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto update_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

  CastShapeSizeToLong(input_shape, &input_shape_);
  CastShapeSizeToLong(update_shape, &update_shape_);
  CastShapeSizeToLong(output_shape, &output_shape_);

  auto begin = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  auto end = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  CheckSliceValid(begin, end, stride, input_shape_);

  data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto dim_offset = CalDimOffset(input_shape_);
  auto type_size = abstract::TypeIdSize(data_type_);
  offset_ = CalOffset(begin, end, dim_offset) * type_size;
  copy_size_ = GetCopySize(dim_offset, begin, end) * type_size;
}

bool TensorCopySlicesCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /* workspace */,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTensorCopySlicesInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTensorCopySlicesOutputsNum, kernel_name_);

  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->addr);
  auto update_addr = reinterpret_cast<uint8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->addr);

  auto ret = memcpy_s(output_addr, outputs[0]->size, input_addr, inputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy input failed. Error no: " << ret;
  }
  ret = memcpy_s(output_addr + offset_, copy_size_, update_addr, copy_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy update failed. Error no: " << ret;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorCopySlices, TensorCopySlicesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
