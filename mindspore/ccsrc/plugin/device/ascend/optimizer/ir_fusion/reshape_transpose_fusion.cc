/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/reshape_transpose_fusion.h"
#include <memory>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckShapeDimInfo(const std::vector<size_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  constexpr auto kShapeSize1 = 1;
  constexpr auto kShapeSize2 = 2;
  if (shape.size() == kShapeSize1 && shape[0] % kCubeSize != 0) {
    return false;
  }
  return !(shape.size() >= kShapeSize2 &&
           (shape[shape.size() - 1] % kCubeSize != 0 || shape[shape.size() - kShapeSize2] % kCubeSize != 0));
}
}  // namespace

const BaseRef ReshapeTransposeFusion::DefinePattern() const {
  const auto prim_reshape = std::make_shared<Primitive>(prim::kPrimReshape->name());
  VectorRef reshape({prim_reshape, input_varptr_});

  return VectorRef({prim::kPrimTranspose, reshape});
}

const AnfNodePtr ReshapeTransposeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto transpose_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kBackendReshapeInputTensorNum);
  MS_EXCEPTION_IF_NULL(transpose_cnode);
  auto reshape_cnode = CheckAnfNodeIfCNodeAndInputSize(transpose_cnode->input(1), kBackendReshapeInputTensorNum);
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  if (common::AnfAlgo::IsDynamicShape(transpose_cnode) || common::AnfAlgo::IsDynamicShape(reshape_cnode)) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr &&
      (kernel_graph->IsInternalOutput(reshape_cnode, 0) || kernel_graph->IsInternalOutput(transpose_cnode, 0))) {
    return nullptr;
  }
  std::vector<size_t> reshape_input0_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(reshape_cnode, 0);
  std::vector<size_t> transpose_output0_shape = common::AnfAlgo::GetOutputInferShape(transpose_cnode, 0);
  if (!CheckShapeDimInfo(reshape_input0_shape) || !CheckShapeDimInfo(transpose_output0_shape)) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kConfusionTransposeDOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), utils::cast<AnfNodePtr>((*equiv)[input_varptr_])};
  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());

  common::AnfAlgo::CopyNodeAttrs(reshape_cnode, new_node);
  common::AnfAlgo::CopyNodeAttr(kAttrPerm, transpose_cnode, new_node);
  common::AnfAlgo::SetNodeAttr(kAttrTransposeFirst, MakeValue(false), new_node);
  auto reshape_output_shape = common::AnfAlgo::GetOutputInferShape(reshape_cnode, 0);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(Convert2Long(reshape_output_shape)), new_node);

  return new_node;
}
}  // namespace opt
}  // namespace mindspore
