/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_launch_transdata.h"

#include <algorithm>
#include "abstract/utils.h"
#include "backend/common/session/single_kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::device::ascend {
void AscendLaunchTransData::FreeDeviceMem(void *addr) { AscendLaunchKernel::FreeDeviceMem(addr); }

size_t AscendLaunchTransData::AlignSizeForLaunchKernel(size_t size) {
  return AscendLaunchKernel::AlignSizeForLaunchKernel(size);
}

uint8_t *AscendLaunchTransData::AllocDeviceMem(size_t size) { return AscendLaunchKernel::AllocDeviceMem(size); }

void AscendLaunchTransData::KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  AscendLaunchKernel::KernelSelect(kernel_graph);
}

void AscendLaunchTransData::KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  AscendLaunchKernel::KernelBuild(kernel_graph);
}

void AscendLaunchTransData::LaunchOpKernel() {
  if (transdata_graph_ == nullptr) {
    // construct transdata kernel graph and set attr
    ConstructKernelGraphAndSetAttr();
    // kernel build
    KernelBuild(transdata_graph_);
  }
  // obtain kernel_mod
  if (transdata_graph_->execution_order().size() != 1) {
    MS_LOG(ERROR) << "The execution order of the transdata graph should have only one node";
    return;
  }
  kernel_mod_ = AnfAlgo::GetKernelMod(transdata_graph_->execution_order()[0]);
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // obtain kernel inputs
  std::vector<kernel::AddressPtr> kernel_inputs;
  auto input = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input);
  input->addr = input_addr_;
  MS_EXCEPTION_IF_NULL(input->addr);
  input->size = total_size_;
  kernel_inputs.push_back(input);
  // obtain kernel outputs
  auto kernel_outputs = ObtainKernelOutputs(kernel_mod_->GetOutputSizeList());
  // obtain kernel workspaces
  auto kernel_workspace = ObtainKernelWorkspaces(kernel_mod_->GetWorkspaceSizeList());
  // launch
  auto ret_status = kernel_mod_->Launch(kernel_inputs, kernel_workspace, kernel_outputs, stream_);
  if (!ret_status) {
    MS_LOG(EXCEPTION) << "Launch transdata single kernel failed";
  }
}

void AscendLaunchTransData::FreeLaunchDeviceMem() {
  input_addr_ = nullptr;
  FreeOutputAndWorkspaceDeviceMem();
}

std::shared_ptr<session::KernelGraph> AscendLaunchTransData::ObtainTransDataKernelGraph() {
  std::vector<TypeId> input_dtypes = {dtype_};
  std::vector<TypeId> output_dtypes = {dtype_};
  // obtain input & output shape
  std::vector<int64_t> input_shape;
  std::transform(shape_.begin(), shape_.end(), std::back_inserter(input_shape), SizeToLong);
  std::vector<std::vector<int64_t>> input_shapes = {{input_shape}};
  std::vector<std::vector<size_t>> output_shapes = {{shape_}};
  auto transdata_graph = session::SingleKernelGraph::ConstructKernelGraphBasedOnSingleOp(
    kTransDataOpName, input_dtypes, input_shapes, output_dtypes, output_shapes);
  MS_EXCEPTION_IF_NULL(transdata_graph);
  return transdata_graph;
}

void AscendLaunchTransData::ConstructKernelGraphAndSetAttr() {
  // construct transdata kernel graph
  transdata_graph_ = ObtainTransDataKernelGraph();
  MS_EXCEPTION_IF_NULL(transdata_graph_);
  // set transdata attr
  if (!transdata_graph_->execution_order().empty()) {
    auto transdata_node = transdata_graph_->execution_order()[0];
    // set output infer type and shape
    common::AnfAlgo::SetOutputInferTypeAndShape({dtype_}, {shape_}, transdata_node.get());
    // set build info
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetKernelType(KernelType::TBE_KERNEL);
    std::vector<TypeId> device_type = {dtype_};
    builder->SetInputsDeviceType(device_type);
    builder->SetOutputsDeviceType(device_type);
    std::vector<std::string> inputs_format = {src_format_};
    std::vector<std::string> outputs_format = {dst_format_};
    builder->SetInputsFormat(inputs_format);
    builder->SetOutputsFormat(outputs_format);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), transdata_node.get());
    // set attr
    common::AnfAlgo::SetNodeAttr(kAttrSrcFormat, MakeValue(src_format_), transdata_node);
    common::AnfAlgo::SetNodeAttr(kAttrDstFormat, MakeValue(dst_format_), transdata_node);
    common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups_), transdata_node);
    common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups_), transdata_node);
  }
}
}  // namespace mindspore::device::ascend
