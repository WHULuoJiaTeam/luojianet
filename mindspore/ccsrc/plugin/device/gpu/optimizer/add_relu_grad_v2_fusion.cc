/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/add_relu_grad_v2_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    inputs_format.push_back(kOpFormat_DEFAULT);
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(node, output_index));
    outputs_format.push_back(kOpFormat_DEFAULT);
  }
  builder.SetInputsDeviceType(inputs_type);
  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsDeviceType(outputs_type);
  builder.SetOutputsFormat(outputs_format);
  return builder.Build();
}
}  // namespace

const BaseRef AddReluGradV2Fusion::DefinePattern() const {
  VectorRef relu_grad = VectorRef({prim::kPrimReluGradV2, VectorRef({prim::kPrimAdd, x1_, x2_}), mask_});
  return relu_grad;
}

const AnfNodePtr AddReluGradV2Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto mask = utils::cast<AnfNodePtr>((*equiv)[mask_]);
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(mask);

  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tensor_add);
  auto users = GetRealNodeUsedList(graph, tensor_add);
  if (users->size() > 1) {
    return nullptr;
  }

  std::vector<size_t> shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  std::vector<size_t> shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  if (shape1 != shape2) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kFusedAddReluGradV2Name);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, mask};
  auto add_relugrad = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_relugrad);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_relugrad.get());
  add_relugrad->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(add_relugrad);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_relugrad.get());
  return add_relugrad;
}
}  // namespace opt
}  // namespace mindspore
