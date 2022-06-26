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
#include "plugin/device/ascend/optimizer/ir_fission/bce_with_logits_loss_fission.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr BCEWithLogitsLossFission::AddReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Copy a new sigmoid node, shape of output is the same as input
  std::vector<AnfNodePtr> new_simoid_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBCEWithLogitsLoss->name()))};
  (void)new_simoid_inputs.insert(new_simoid_inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  CNodePtr new_cnode = NewCNode(new_simoid_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  auto predict_input = cnode->inputs()[kIndex1];
  auto new_node_dtype = {common::AnfAlgo::GetOutputInferDataType(predict_input, 0)};
  auto new_node_shape = {common::AnfAlgo::GetOutputDetailShape(predict_input, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(new_node_dtype, new_node_shape, new_cnode.get());

  // Add reduce node
  string reduction = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrReduction);
  MS_LOG(INFO) << "Create reduce node, reduction attr is: " << reduction;
  std::vector<AnfNodePtr> reduce_inputs;
  if (reduction == "sum") {
    reduce_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())), new_cnode};
  } else if (reduction == "mean") {
    reduce_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceMean->name())), new_cnode};
  } else {
    MS_LOG(INFO) << "Reduction attr is not mean or sum, can not do fission.";
    return nullptr;
  }
  auto reduce_node = NewCNode(reduce_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(reduce_node);
  auto type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (type == kNumberTypeFloat16) {
    type = kNumberTypeFloat32;
  }
  auto shape = {common::AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape({type}, shape, reduce_node.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{}), reduce_node);
  common::AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_node);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_node);
  reduce_node->set_scope(cnode->scope());
  return reduce_node;
}

const BaseRef BCEWithLogitsLossFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimBCEWithLogitsLoss, Xs});
}

const AnfNodePtr BCEWithLogitsLossFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (GetBoolAttr(cnode, kAttrVisited)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  if (cnode->inputs().size() == 0) {
    return nullptr;
  }
  if (!common::AnfAlgo::HasNodeAttr("reduction", cnode)) {
    MS_LOG(INFO) << "Has no reduction attr.";
    return nullptr;
  }
  return AddReduceNode(func_graph, node);
}
}  // namespace opt
}  // namespace mindspore
