/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/pynative/op_runtime_info.h"

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::runtime {
namespace {
void CacheForExecutionOrder(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &nodes = graph->execution_order();
  for (auto const &node : nodes) {
    std::vector<std::string> formats;
    std::vector<TypeId> types;
    std::vector<size_t> tensor_sizes;
    auto output_num = common::AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      std::string output_format = AnfAlgo::GetOutputFormat(node, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(node, i);
      formats.emplace_back(output_format);
      types.emplace_back(output_type);
      tensor_sizes.emplace_back(tensor_size);
    }

    // For input
    std::vector<std::pair<device::KernelInfo *, size_t>> input_kernel_infos;
    auto input_size = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_size; ++i) {
      session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i, true);
      MS_EXCEPTION_IF_NULL(kernel_with_index.first);
      input_kernel_infos.emplace_back(dynamic_cast<device::KernelInfo *>(kernel_with_index.first->kernel_info()),
                                      kernel_with_index.second);
    }

    // For workspace and output
    MS_EXCEPTION_IF_NULL(node);
    auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());

    node->set_user_data<runtime::OpRuntimeInfo>(
      std::make_shared<runtime::OpRuntimeInfo>(formats, types, tensor_sizes, kernel_info, input_kernel_infos));
  }
}

void CacheForGraphInputs(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &inputs = graph->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<Parameter>()) {
      continue;
    }
    std::vector<std::string> formats;
    std::vector<TypeId> types;
    std::vector<size_t> tensor_sizes;
    auto output_size = common::AnfAlgo::GetOutputTensorNum(input);
    for (size_t index = 0; index < output_size; index++) {
      auto format = AnfAlgo::GetOutputFormat(input, index);
      auto type_id = AnfAlgo::GetOutputDeviceDataType(input, index);
      if (type_id == kTypeUnknown) {
        type_id = common::AnfAlgo::GetOutputInferDataType(input, index);
      }
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(input, index);
      formats.emplace_back(format);
      types.emplace_back(type_id);
      tensor_sizes.emplace_back(tensor_size);
    }
    input->set_user_data<runtime::OpRuntimeInfo>(std::make_shared<runtime::OpRuntimeInfo>(
      formats, types, tensor_sizes, nullptr, std::vector<std::pair<device::KernelInfo *, size_t>>()));
  }
}
}  // namespace

std::string OpRuntimeInfo::output_format(size_t index) const {
  if (index >= output_format_.size()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << index << " total output_format:" << output_format_.size();
  }
  return output_format_[index];
}

TypeId OpRuntimeInfo::output_type(size_t index) const {
  if (index >= output_type_.size()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << index << " total output_type:" << output_type_.size();
  }
  return output_type_[index];
}

size_t OpRuntimeInfo::output_tensor_size(size_t index) const {
  if (index >= output_tensor_size_.size()) {
    MS_LOG(EXCEPTION) << "Invalid index::" << index << " total output_tensor_size:" << output_tensor_size_.size();
  }
  return output_tensor_size_[index];
}

device::DeviceAddressPtr OpRuntimeInfo::GetOutputDeviceAddress(size_t index) const {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  return kernel_info_->GetMutableOutputAddr(index);
}

device::DeviceAddressPtr OpRuntimeInfo::GetWorkspaceDeviceAddress(size_t index) const {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  return kernel_info_->GetMutableWorkspaceAddr(index);
}

device::DeviceAddressPtr OpRuntimeInfo::GetInputDeviceAddress(size_t index) const {
  if (index >= input_kernel_infos_.size()) {
    MS_LOG(ERROR) << "Output range! index:" << index << " input size:" << input_kernel_infos_.size();
    return nullptr;
  }

  auto kernel_info_pair = input_kernel_infos_[index];
  MS_EXCEPTION_IF_NULL(kernel_info_pair.first);
  return kernel_info_pair.first->GetMutableOutputAddr(kernel_info_pair.second);
}

size_t OpRuntimeInfo::GetInputSize() const { return input_kernel_infos_.size(); }

size_t OpRuntimeInfo::GetOutputSize() const {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  return kernel_info_->output_address_list().size();
}

size_t OpRuntimeInfo::GetWorkspaceSize() const {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  return kernel_info_->workspace_address_list().size();
}

void OpRuntimeInfo::CacheGraphOpRuntimeInfo(const KernelGraphPtr &graph) {
  CacheForExecutionOrder(graph);
  CacheForGraphInputs(graph);
}
}  // namespace mindspore::runtime
