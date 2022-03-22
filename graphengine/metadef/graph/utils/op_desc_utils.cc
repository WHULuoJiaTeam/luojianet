/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#include "graph/utils/op_desc_utils.h"
#include <algorithm>
#include <memory>
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/op_desc_impl.h"
#include "graph/operator.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/operator_impl.h"


/*lint -e512 -e737 -e752*/
namespace ge {
const char_t OP_DESC_QUANT_PARAMS[] = "quantize_factor";

namespace {
const int32_t CONST_OP_NORMAL_WEIGHT_SIZE = 1;
}

bool OpDescUtils::ClearInputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return false, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, REPORT_INNER_ERROR("E19999", "opdesc is nullptr.");
                   return false, "[Check][Param] opdesc is nullptr");
  std::vector<int32_t> index_list;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      index_list.push_back(in_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's in anchor index need shrink
  if (node->GetOpDesc()->impl_ == nullptr) {
    GELOGE(FAILED, "[Clear][InputDesc] Op desc impl is nullptr. ");
    return false;
  }
  for (size_t i = 0UL; i < index_list.size(); ++i) {
    const auto iter = node->GetOpDesc()->impl_->inputs_desc_.begin() + static_cast<int64_t>(index_list[i]);
    if (iter < node->GetOpDesc()->impl_->inputs_desc_.end()) {
      (void)node->GetOpDesc()->impl_->inputs_desc_.erase(iter);
    } else {
      GELOGW("[Clear][InputDesc] inputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearInputDesc(const OpDescPtr op_desc,
                                                                                const uint32_t index) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr && op_desc->impl_ != nullptr,
                   REPORT_INNER_ERROR("E19999", "op_desc is nullptr, check invalid");
                   return false, "[Check][Param] op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->impl_->inputs_desc_.size(),
                   REPORT_INNER_ERROR("E19999", "index %u is invalid, out of range(0, %zu).",
                                      index, op_desc->impl_->inputs_desc_.size());
                   return false,
                   "[Check][Param] index %u is invalid, out of range(0, %zu).",
                   index, op_desc->impl_->inputs_desc_.size());

  const auto iter = op_desc->impl_->inputs_desc_.begin() + static_cast<int64_t>(index);
  if (iter < op_desc->impl_->inputs_desc_.end()) {
    (void)op_desc->impl_->inputs_desc_.erase(iter);
  } else {
    GELOGW("[Clear][InputDesc] inputs_desc_ iterator out of range.");
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::HasQuantizeFactorParams(const OpDescPtr &op_desc) {
  GE_CHK_BOOL_EXEC_INFO(op_desc != nullptr, return false, "op_desc is nullptr");
  return op_desc->HasAttr(OP_DESC_QUANT_PARAMS);
}

bool OpDescUtils::ClearOutputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "node is nullptr, check invalid.");
                   return false, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, REPORT_INNER_ERROR("E19999", "opdesc is nullptr.");
                   return false, "[Check][Param] opdesc is nullptr");
  std::vector<int32_t> index_list;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor->GetPeerInDataAnchors().empty()) {
      index_list.push_back(out_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's out anchor index need shrink
  if (node->GetOpDesc()->impl_ == nullptr) {
    GELOGE(FAILED, "[Clear][OutputDesc] Op desc impl is nullptr. ");
    return false;
  }
  for (size_t i = 0UL; i < index_list.size(); ++i) {
    const auto iter = node->GetOpDesc()->impl_->outputs_desc_.begin() + static_cast<int64_t>(index_list[i]);
    if (iter < node->GetOpDesc()->impl_->outputs_desc_.end()) {
      (void)node->GetOpDesc()->impl_->outputs_desc_.erase(iter);
    } else {
      GELOGW("[Clear][OutputDesc] outputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearOutputDesc(const OpDescPtr &op_desc,
                                                                                 const uint32_t index) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr && op_desc->impl_ != nullptr,
                   REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid");
                   return false, "[Check][Param] op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->impl_->outputs_desc_.size(),
                   REPORT_INNER_ERROR("E19999", "index %u is invalid. out of range(0, %zu)",
                                      index, op_desc->impl_->outputs_desc_.size());
                   return false,
                   "[Check][Param] index %u is invalid. out of range(0, %zu)",
                   index, op_desc->impl_->outputs_desc_.size());
  const auto iter = op_desc->impl_->outputs_desc_.begin() + static_cast<int64_t>(index);
  if (iter < op_desc->impl_->outputs_desc_.end()) {
    (void)op_desc->impl_->outputs_desc_.erase(iter);
  } else {
    GELOGW("[Clear][OutputDesc] outputs_desc_ iterator out of range.");
  }
  return true;
}

bool OpDescUtils::HasQuantizeFactorParams(const OpDesc &op_desc) { return op_desc.HasAttr(OP_DESC_QUANT_PARAMS); }

GeTensorPtr OpDescUtils::MutableWeights(OpDesc &op_desc) {
  GeTensorPtr weight = nullptr;
  (void)AttrUtils::MutableTensor(&op_desc, ATTR_NAME_WEIGHTS, weight);
  return weight;
}

GE_FUNC_HOST_VISIBILITY GeTensorPtr OpDescUtils::MutableWeights(const OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] op_desc is null");
    return nullptr;
  }
  return MutableWeights(*op_desc);
}

graphStatus OpDescUtils::SetWeights(OpDesc &op_desc, const GeTensorPtr weight) {
  if (weight == nullptr) {
    REPORT_INNER_ERROR("E19999", "weight is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] weight is null");
    return GRAPH_FAILED;
  }
  return AttrUtils::SetTensor(&op_desc, ATTR_NAME_WEIGHTS, weight) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus OpDescUtils::SetWeights(OpDescPtr op_desc, const GeTensorPtr weight) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(weight);
  return SetWeights(*op_desc, weight);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<ConstGeTensorPtr> OpDescUtils::GetWeights(const ge::Node &node) {
  auto weights = MutableWeights(node);
  std::vector<ConstGeTensorPtr> ret(weights.size());
  (void)std::copy(weights.begin(), weights.end(), ret.begin());
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ConstGeTensorPtr> OpDescUtils::GetWeights(
    const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return std::vector<ge::ConstGeTensorPtr>();
  }
  return GetWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::NodePtr> OpDescUtils::GetConstInputNode(
    const ge::Node &node) {
  std::vector<ge::NodePtr> ret;
  const auto in_anchors = node.GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      // normally out_anchor could be null, this is ok
      GELOGD("node %s' peer_out_anchor is null", node.GetName().c_str());
      continue;
    }
    auto in_node = out_anchor->GetOwnerNode();
    while (true) {
      if (in_node == nullptr) {
        break;
      }
      if (ConstantUtils::IsConstant(in_node)) {
        ret.push_back(in_node);
        break;
      } else if (in_node->GetType() == DATA) {
        if (NodeUtils::IsWhileVaryingInput(in_node)) {
          break;
        }
        in_node = NodeUtils::GetParentInput(in_node);
      } else if ((in_node->GetType() == ENTER) || (in_node->GetType() == REFENTER)) {
        bool is_constant = false;
        (void)AttrUtils::GetBool(in_node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant);
        if (!is_constant) {
          break;
        }
        // Enter node has and only has one input
        if (in_node->GetInDataNodes().size() != 1U) {
          GELOGW("[Get][ConstInput] Check number of input_nodes for Enter node %s failed, input_node_num=%zu.",
                 in_node->GetName().c_str(), in_node->GetInDataNodes().size());
          break;
        }
        in_node = in_node->GetInDataNodes().at(0UL);
      } else {
        break;
      }
    }
  }
  return ret;
}

std::vector<NodeToOutAnchor> OpDescUtils::GetConstInputNodeAndAnchor(const ge::Node &node) {
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> ret;
  auto in_nodes_and_anchors = node.GetInDataNodesAndAnchors();
  for (const auto &in_node_2_anchor : in_nodes_and_anchors) {
    auto in_node = in_node_2_anchor.first;
    auto in_node_2_out_anchor = in_node_2_anchor;
    while (true) {
      if (in_node == nullptr) {
        break;
      }
      if (ConstantUtils::IsConstant(in_node)) {
        ret.push_back(in_node_2_out_anchor);
        break;
      } else if (in_node->GetType() == DATA) {
        if (NodeUtils::IsWhileVaryingInput(in_node)) {
          break;
        }
        in_node_2_out_anchor = NodeUtils::GetParentInputAndAnchor(in_node);
        in_node = in_node_2_out_anchor.first;
      } else if ((in_node->GetType() == ENTER) || (in_node->GetType() == REFENTER)) {
        bool is_constant = false;
        (void)AttrUtils::GetBool(in_node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant);
        if (!is_constant) {
          break;
        }
        // Enter node has and only has one input
        if (in_node->GetInDataNodes().size() != 1) {
          GELOGW("[Get][ConstInput] Check number of input_nodes for Enter node %s failed, input_node_num=%zu.",
                 in_node->GetName().c_str(), in_node->GetInDataNodes().size());
          break;
        }
        auto peer_out_anchor = in_node->GetInDataAnchor(0)->GetPeerOutAnchor();
        in_node = peer_out_anchor->GetOwnerNode();
        in_node_2_out_anchor = std::make_pair(in_node, peer_out_anchor);
      } else {
        break;
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ConstGeTensorPtr> OpDescUtils::GetInputData(
    const std::vector<ge::NodePtr> &input_nodes) {
  std::vector<ConstGeTensorPtr> ret;

  for (const auto &input_node : input_nodes) {
    const auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      REPORT_CALL_ERROR("E19999", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<ConstGeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

vector<ConstGeTensorPtr> OpDescUtils::GetWeightsFromNodes(
    const std::vector<NodeToOutAnchor> &input_nodes_2_out_anchors) {
  std::vector<ConstGeTensorPtr> ret;
  for (const auto &input_node_2_anchor : input_nodes_2_out_anchors) {
    const auto input_node = input_node_2_anchor.first;
    GeTensorPtr temp_weight ;
    (void)ConstantUtils::MutableWeight(input_node->GetOpDesc(),
                                       input_node_2_anchor.second->GetIdx(), temp_weight);
    if (temp_weight == nullptr) {
      REPORT_CALL_ERROR("E19999", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<ConstGeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}
size_t OpDescUtils::GetNonConstInputsSize(const ge::Node &node) {
  if (NodeUtils::IsAnchorStatusSet(node)) {
    size_t input_num = 0UL;
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        input_num++;
        continue;
      }
    }
    return input_num;  // lint !e712
  } else {
    GE_IF_BOOL_EXEC(
        node.GetInDataNodes().size() < GetConstInputs(node).size(),
        REPORT_INNER_ERROR("E19999", "InDataNodes size:%zu is smaller than ConstInputs size:%zu",
                           node.GetInDataNodes().size(), GetConstInputs(node).size());
        GELOGE(GRAPH_FAILED, "[Check][Param] %zu is smaller than %zu",
               node.GetInDataNodes().size(), GetConstInputs(node).size());
        return 0UL);
    return node.GetInDataNodes().size() - GetConstInputs(node).size();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDescUtils::GetNonConstInputsSize(const ge::ConstNodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return 0UL;
  }
  return GetNonConstInputsSize(*node);
}

GeTensorDesc OpDescUtils::GetNonConstInputTensorDesc(const ge::Node &node, const size_t index_non_const) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, REPORT_CALL_ERROR("E19999", "node.GetOpDesc() is nullptr!");
                   return GeTensorDesc(), "[Check][Param] node.GetOpDesc() is nullptr!");
  size_t i = 0UL;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          return node.GetOpDesc()->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto owner_node = peer_anchor->GetOwnerNode();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        return node.GetOpDesc()->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
      }
      ++i;
    }
  }
  return GeTensorDesc();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc
OpDescUtils::GetNonConstInputTensorDesc(const ge::ConstNodePtr &node, const size_t index_non_const) {
  CHECK_FALSE_EXEC(node != nullptr, return GeTensorDesc());
  return GetNonConstInputTensorDesc(*node, index_non_const);
}

bool OpDescUtils::GetNonConstInputIndex(const ge::Node &node, const size_t index_non_const, size_t &index) {
  bool ret = false;
  size_t i = 0UL;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          index = static_cast<size_t>(anchor->GetIdx());
          ret = true;
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto owner_node = peer_anchor->GetOwnerNode();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        index = static_cast<size_t>(anchor->GetIdx());
        ret = true;
      }
      ++i;
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::GetNonConstInputIndex(const ge::ConstNodePtr &node,
                                                                                       const size_t index_non_const,
                                                                                       size_t &index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return GetNonConstInputIndex(*node, index_non_const, index);
}

bool OpDescUtils::IsNonConstInput(const ge::Node &node, const size_t index) {
  bool ret = false;
  if (index < node.GetAllInDataAnchors().size()) {
    if (NodeUtils::IsAnchorStatusSet(node)) {
      ret = (ge::AnchorUtils::GetStatus(node.GetInDataAnchor(static_cast<int>(index))) == ANCHOR_DATA);  // lint !e712
    } else {
      for (const auto &anchor : node.GetAllInDataAnchors()) {
        if (anchor->GetIdx() != static_cast<int>(index)) {
          continue;
        }
        auto peer_anchor = anchor->GetPeerOutAnchor();
        if (peer_anchor == nullptr) {
          break;
        }
        auto owner_node = peer_anchor->GetOwnerNode();
        if (owner_node == nullptr) {
          break;
        }
        ret = (owner_node->GetType() != CONSTANT);
      }
    }
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::IsNonConstInput(const ge::ConstNodePtr &node,
                                                                                 const size_t index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return IsNonConstInput(*node, index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::NodePtr> OpDescUtils::GetConstInputs(
    const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return std::vector<ge::NodePtr>();
  }
  return GetConstInputs(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::GeTensorDesc> OpDescUtils::GetNonConstTensorDesc(
    const ge::ConstNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return std::vector<ge::GeTensorDesc>();
  }
  std::vector<ge::GeTensorDesc> ret;
  if (NodeUtils::IsAnchorStatusSet(*node)) {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(in_anchor) == ANCHOR_DATA) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(in_anchor->GetIdx()));
      }
    }
  } else {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr || out_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
        continue;
      }
      if (out_anchor->GetOwnerNode()->GetOpDesc()->GetType() != CONSTANT) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(in_anchor->GetIdx()));
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<ge::NodePtr> OpDescUtils::GetConstInputs(const ge::Node &node) {
  std::vector<ge::NodePtr> ret;
  auto in_anchors = node.GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) continue;

    auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == CONSTANT) {
      ret.push_back(in_node);
    } else if (in_node->GetType() == SWITCH && node.GetType() == MATMUL) {
      // const --> switch --> matmul
      auto switch_input = GetConstInputs(*in_node);
      if (switch_input.size() > 0U) {
        (void)ret.insert(ret.end(), switch_input.begin(), switch_input.end());
      }
    } else if (in_node->GetType() == DATA) {
      const auto parent = NodeUtils::GetParentInput(in_node);
      if ((parent != nullptr) && (parent->GetType() == CONSTANT)) {
        ret.push_back(parent);
      }
    } else {
      // do nothing
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::Node &node) {
  std::vector<GeTensorPtr> ret;
  auto op_desc = node.GetOpDesc();
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERROR("E19999", "param node's op_desc is nullptr.");
                   return ret, "[Check][Param] op_desc is nullptr!");
  // Place holder operator, try to get the weight from parent node
  // when parent node is const operator
  if (node.GetType() == PLACEHOLDER) {
    std::string parent_op;
    (void) AttrUtils::GetStr(op_desc, "parentOpType", parent_op);
    // This if judgment is necessary because the current subgraph optimization is multithreaded
    // and the parent node of the PLD operation should be a stable type, such as const
    if (parent_op == CONSTANT || parent_op == CONSTANTOP) {
      NodePtr parent_node = nullptr;
      parent_node = op_desc->TryGetExtAttr("parentNode", parent_node);
      if (parent_node != nullptr) {
        op_desc = parent_node->GetOpDesc();
        GELOGD("pld[%s] get weight from const[%s]", node.GetName().c_str(), op_desc->GetName().c_str());
      }
    }
  }
  // Const operator, take the weight directly
  // In some case, Placeholder operator may has it's peer const node's weight
  if ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP) || (op_desc->GetType() == PLACEHOLDER)) {
    const auto weight = MutableWeights(op_desc);
    if (weight == nullptr) {
      GELOGD("op type %s has no weight, op name:%s", node.GetType().c_str(), node.GetName().c_str());
      return ret;
    }
    ret.push_back(weight);
    return ret;
  }

  if (node.GetType() == DATA) {
    const auto parent = NodeUtils::GetParentInput(node);
    if ((parent != nullptr) && NodeUtils::IsConst(*parent)) {
      const auto weight = MutableWeights(parent->GetOpDesc());
      if (weight == nullptr) {
        GELOGI("const op has no weight, op name:%s", parent->GetName().c_str());
        return ret;
      }
      ret.push_back(weight);
    }
    return ret;
  }

  // Other operators, get weights from connected constop
  const auto input_nodes = GetConstInputs(node);
  for (const auto &input_node : input_nodes) {
    const auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      REPORT_INNER_ERROR("E19999", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<GeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::NodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return std::vector<ge::GeTensorPtr>();
  }
  return MutableWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const std::vector<ge::GeTensorPtr> &weights) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, REPORT_CALL_ERROR("E19999", "opdesc of node is nullptr.");
                   return GRAPH_PARAM_INVALID, "[Check][Param] node.GetOpDesc is nullptr!");
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights[0UL]);
    }
    GELOGI("const op weight size %zu should be 1", weights.size());
    return GRAPH_PARAM_INVALID;
  }

  const auto input_nodes = GetConstInputs(node);
  if (weights.size() < input_nodes.size()) {
    REPORT_INNER_ERROR("E19999", "weights count:%zu can't be less than const input count:%zu, node:%s(%s)",
                       weights.size(), input_nodes.size(), node.GetName().c_str(), node.GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] weights count:%zu can't be less than const input count:%zu",
           weights.size(), input_nodes.size());
    return GRAPH_PARAM_INVALID;
  }

  ge::NamedAttrs named_attrs;
  (void)ge::AttrUtils::SetListTensor(named_attrs, "key", weights);
  std::vector<ge::GeTensorPtr> copy_weights;
  (void)ge::AttrUtils::MutableListTensor(named_attrs, "key", copy_weights);

  for (size_t i = 0UL; i < input_nodes.size(); ++i) {
    if (input_nodes[i]->GetOpDesc() != nullptr) {
      SetWeights(input_nodes[i]->GetOpDesc(), copy_weights[i]);
    }
  }

  // If set more weights than constop, need to add constop
  for (size_t i = input_nodes.size(); i < copy_weights.size(); ++i) {
    // Use org weight before SetWeights Overwrite
    const auto const_opdesc = CreateConstOp(copy_weights[i]);
    GE_CHECK_NOTNULL(const_opdesc);

    const auto owner_graph = node.GetOwnerComputeGraph();
    if (owner_graph == nullptr) {
      REPORT_CALL_ERROR("E19999", "node's graph is empty, node name: %s", node.GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Graph] node's graph is empty, name: %s", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    const auto const_node = owner_graph->AddNodeFront(const_opdesc);
    GE_CHK_BOOL_EXEC(node.AddLinkFrom(const_node) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "node:%s add link failed.", node.GetName().c_str());
                     GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] graph add link failed! node:%s",
                            node.GetName().c_str());
                     return GRAPH_FAILED);
    const std::vector<ge::NodePtr> original_nodes;
    ge::GraphUtils::RecordOriginalNames(original_nodes, const_node);
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const std::map<int, ge::GeTensorPtr> &weights_map) {
  GE_CHECK_NOTNULL(node.GetOpDesc());
  // 1. node is const
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights_map.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights_map.begin()->second);
    }
    REPORT_INNER_ERROR("E19999", "const op %s weight size %zu should be 1", node.GetName().c_str(), weights_map.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] const op %s weight size %zu should be 1",
           node.GetName().c_str(), weights_map.size());
    return GRAPH_PARAM_INVALID;
  }
  // 2. node is not const
  for (const auto &pair:weights_map) {
    const auto idx = pair.first;
    // idx = in data anchor size is valid, it meant to add a new const node
    if ((idx < 0) || (static_cast<size_t>(idx) > node.GetAllInDataAnchorsSize())) {
      REPORT_CALL_ERROR("E19999", "Invalid map key: %d of node[%s].", idx, node.GetName().c_str());
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid map key: %d of node[%s].", idx, node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    const auto peer_node = NodeUtils::GetInDataNodeByIndex(node, idx);
    if (peer_node != nullptr) {
      // a. update const input node
      if (peer_node->GetType() != CONSTANT) {
        REPORT_INNER_ERROR("E19999", "op %s [%d]'s input node should be const, but is %s type:%s ",
                           node.GetName().c_str(), pair.first,
                           peer_node->GetName().c_str(), peer_node->GetType().c_str());
        GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] op %s [%d]'s input node should be const, but is %s type:%s ",
               node.GetName().c_str(), pair.first, peer_node->GetName().c_str(), peer_node->GetType().c_str());
      }
      SetWeights(peer_node->GetOpDesc(), pair.second);
    } else {
      // b. create new const input node
      const auto const_opdesc = CreateConstOp(pair.second);
      GE_CHECK_NOTNULL(const_opdesc);
      const auto owner_graph = node.GetOwnerComputeGraph();
      if (owner_graph == nullptr) {
        REPORT_CALL_ERROR("E19999", "node's graph is empty, node name: %s", node.GetName().c_str());
        GELOGE(GRAPH_PARAM_INVALID, "[Get][Graph] node's graph is empty, name: %s", node.GetName().c_str());
        return GRAPH_PARAM_INVALID;
      }
      const auto const_node = owner_graph->AddNodeFront(const_opdesc);
      if (node.AddLinkFrom(static_cast<const uint32_t>(pair.first), const_node) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "op %s add const to input index[%d] failed", node.GetName().c_str(), pair.first);
        GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] op %s add const to input index[%d] failed",
               node.GetName().c_str(), pair.first);
        return GRAPH_FAILED;
      }
    }
  }
  NodeUtils::UpdateIsInputConst(node);
  return GRAPH_SUCCESS;
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr) {
  GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, REPORT_INNER_ERROR("E19999", "tensor_ptr is nullptr, check invalid.");
                   return nullptr, "[Check][Param] tensor_ptr is nullptr!");
  const shared_ptr<OpDesc> const_opdesc = ComGraphMakeShared<OpDesc>();
  if (const_opdesc == nullptr) {
    REPORT_CALL_ERROR("E19999", "create OpDesc failed.");
    GELOGE(GRAPH_FAILED, "[Create][OpDesc] failed to make_shared ");
    return nullptr;
  }

  CHECK_FALSE_EXEC(SetWeights(const_opdesc, tensor_ptr) == ge::GRAPH_SUCCESS, return nullptr);

  const_opdesc->SetType(CONSTANT);

  thread_local int64_t const_count = 0;
  const_opdesc->SetName("dynamic_const_" + std::to_string(GeLog::GetTid()) + "_" + std::to_string(const_count));
  GELOGI("add const op: %s", const_opdesc->GetName().c_str());
  ++const_count;

  (void)const_opdesc->AddOutputDesc(tensor_ptr->GetTensorDesc());

  GELOGI("after add const op: %s", const_opdesc->GetName().c_str());

  return const_opdesc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::AddConstOpToAnchor(const InDataAnchorPtr in_anchor, const GeTensorPtr &tensor_ptr) {
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(tensor_ptr);
  const auto const_opdesc = CreateConstOp(tensor_ptr);
  GE_CHECK_NOTNULL(const_opdesc);
  const auto in_node = in_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(in_node);
  const auto owner_graph = in_node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "node's graph is empty, name: %s", in_node->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "[Get][Graph] node's graph is empty, name: %s", in_node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto const_node = in_node->GetOwnerComputeGraph()->AddNodeFront(const_opdesc);
  GE_CHECK_NOTNULL(const_node);
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddEdge const %s to node %s failed", const_node->GetName().c_str(),
                      in_node->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "[Add][Edge] const %s to node %s failed.", const_node->GetName().c_str(),
           in_node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::NodePtr node, const std::vector<ge::GeTensorPtr> &weights) {
  GE_CHECK_NOTNULL(node);
  return SetWeights(*node, weights);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::ClearWeights(const ge::NodePtr node) {
  GE_CHECK_NOTNULL(node);
  const auto const_ops = GetConstInputs(node);
  const auto graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "GetOwnerComputeGraph failed, graph is nullptr, node:%s", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] Graph is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  for (const auto &const_op : const_ops) {
    GE_CHK_STATUS_RET(GraphUtils::IsolateNode(const_op, {}), "[Isolate][Node] %s, type:%s failed",
                      const_op->GetName().c_str(), const_op->GetType().c_str());
    GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, const_op),
                      "[Remove][Node] %s, type: %s without relink failed", const_op->GetName().c_str(),
                      const_op->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

///
/// @brief Add input
/// @param [in] name
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder& OpDescBuilder::AddInput(const std::string &name) {
  inputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

///
/// @brief Add input
/// @param [in] name
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
OpDescBuilder& OpDescBuilder::AddInput(const std::string &name, const GeTensorDesc &tensor) {
  inputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

///
/// @brief Add dynamic input
/// @param [in] name
/// @param [in] num
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder& OpDescBuilder::AddDynamicInput(const std::string &name,
                                                                                             const uint32_t num) {
  for (uint32_t i = 0U; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

///
/// @brief Add dynamic input
/// @param [in] name
/// @param [in] num
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
OpDescBuilder& OpDescBuilder::AddDynamicInput(const std::string &name, const uint32_t num, const GeTensorDesc &tensor) {
  for (uint32_t i = 0U; i < num; i++) {
    inputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

///
/// @brief Add output
/// @param [in] name
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder& OpDescBuilder::AddOutput(const std::string &name) {
  outputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

///
/// @brief Add output
/// @param [in] name
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
OpDescBuilder& OpDescBuilder::AddOutput(const std::string &name, const GeTensorDesc &tensor) {
  outputs_.emplace_back(std::make_pair(name, tensor));
  return *this;
}

///
/// @brief Add dynamic output
/// @param [in] name
/// @param [in] num
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescBuilder& OpDescBuilder::AddDynamicOutput(const std::string &name,
                                                                                              const uint32_t num) {
  for (uint32_t i = 0U; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), GeTensorDesc()));
  }
  return *this;
}

///
/// @brief Add dynamic output
/// @param [in] name
/// @param [in] num
/// @param [in] tensor
/// @return OpDescBuilder
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
OpDescBuilder& OpDescBuilder::AddDynamicOutput(const std::string &name, const uint32_t num,
                                               const GeTensorDesc &tensor) {
  for (uint32_t i = 0U; i < num; i++) {
    outputs_.emplace_back(std::make_pair(name + std::to_string(i), tensor));
  }
  return *this;
}

///
/// @brief Build op_desc
/// @return OpDescPtr
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescBuilder::Build() {
  OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name_, type_));
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "create opdesc failed, name:%s, type:%s.", name_.c_str(), type_.c_str());
    GELOGE(GRAPH_FAILED, "[Create][OpDesc] failed, name:%s, type:%s.", name_.c_str(), type_.c_str());
    return nullptr;
  }

  for (auto &input : inputs_) {
    if (op_desc->AddInputDesc(input.first, input.second) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddInputDesc failed, op:%s.", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed, op:%s.", name_.c_str());
      return nullptr;
    }
  }

  for (auto &output : outputs_) {
    if (op_desc->AddOutputDesc(output.first, output.second) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddOutputDesc failed, op:%s", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputDesc] failed, op:%s.", name_.c_str());
      return nullptr;
    }
  }

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus OpDescUtils::SetSubgraphInstanceName(const std::string &subgraph_name,
                                                 const std::string &subgraph_instance_name,
                                                 OpDescPtr &op_desc) {
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  const auto iter = subgraph_names_to_index.find(subgraph_name);
  if (iter == subgraph_names_to_index.end()) {
    REPORT_INNER_ERROR("E19999",
                       "Failed to set subgraph instance %s for node %s type %s, the subgraph name %s does not exists",
                       subgraph_instance_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       subgraph_name.c_str());
    GELOGE(GRAPH_PARAM_INVALID,
        "[Check][Param] Failed to set subgraph instance %s for node %s type %s, the subgraph name %s does not exists",
        subgraph_instance_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), subgraph_name.c_str());
    return GRAPH_PARAM_INVALID;
  }

  return op_desc->SetSubgraphInstanceName(iter->second, subgraph_instance_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ConstGeTensorBarePtr OpDescUtils::GetInputConstData(const Operator &op, uint32_t idx) {
  if (op.operator_impl_ == nullptr) {
    GELOGW("[Check][Param] Op(%s) operator_impl_ is nullptr.", op.GetName().c_str());
    return nullptr;
  }

  ConstGeTensorPtr ge_tensor = nullptr;
  if (op.operator_impl_->GetInputConstData(idx, ge_tensor) == GRAPH_SUCCESS) {
    return ge_tensor.get();
  }
  GELOGW("[Get][ConstInput] Op(%s) %u get input const data failed", op.GetName().c_str(), idx);
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void OpDescUtils::SetRuntimeContextToOperator(const Operator &op, RuntimeInferenceContext *context) {
  op.operator_impl_->runtime_context_ = context;
}
}  // namespace ge
/*lint +e512 +e737 +e752*/
