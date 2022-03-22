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

#include "graph/passes/transop_depth_fusion_pass.h"

#include <algorithm>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "common/transop_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
graphStatus TransOpDepthFusionPass::Run(ComputeGraphPtr graph) {
  GELOGI("[TransOpDepthFusionPass]: optimize in depth begin...");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (TransOpUtil::IsTransOp(node)) {
      continue;
    }

    GELOGD("Current normal node is: %s, type: %s, begin in-depth recursive", node->GetName().c_str(),
           node->GetType().c_str());
    for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_anchor);
      for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        if (RecursiveInDepth(peer_in_anchor, graph) != GRAPH_SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Call][RecursiveInDepth] failed, root node is:%s, type:%s", node->GetName().c_str(),
                 node->GetType().c_str());
        }
      }
    }
  }
  GELOGI("[TransOpDepthFusionPass]: Optimize in depth success...");
  return GRAPH_SUCCESS;
}

/// @@ Method:
/// Depth-first recursive strategy was utilized to traverse all the trans ops.
/// Both trans ops will be offset when the back one's output desc is consistent
/// with it's former neighbor's input.
/// @@ Limitation:
/// The current method only judge the neighbors. Trans ops separated by some
/// other ops which can't be offset are not taken into account in current
/// @@ Recursive depth
/// To ensure that the stack does not overflow, the maximum depth in recursive is
/// set to be maxRecursiveDepth = 20. More trans ops are seen abnormally.
graphStatus TransOpDepthFusionPass::RecursiveInDepth(const InDataAnchorPtr &dst_in_anchor,
                                                     const ge::ComputeGraphPtr &graph) {
  static unsigned int temp_depth = 0;
  static const unsigned int max_recursive_depth = 20;
  temp_depth++;
  if (temp_depth >= max_recursive_depth) {
    GELOGI(
        "Caution: recursive depth is become %u."
        "It's abnormally to have so many trans ops between two normal ops"
        "Please check your graph in detail!"
        "The search terminate here and continue to another branch.",
        temp_depth);
    temp_depth--;
    return GRAPH_SUCCESS;
  }

  if (dst_in_anchor == nullptr || dst_in_anchor->GetOwnerNode() == nullptr ||
      dst_in_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param dst_in_anchor related node info has nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Param dst_in_anchor related node info has nullptr.");
    return GRAPH_FAILED;
  }
  auto node = dst_in_anchor->GetOwnerNode();
  if (!TransOpUtil::IsTransOp(node) || dst_in_anchor->GetIdx() != TransOpUtil::GetTransOpDataIndex(node)) {
    GELOGD("Now the end of this branch, node: %s, type: %s, recursive depth: %u", node->GetName().c_str(),
           node->GetType().c_str(), temp_depth);
    temp_depth--;
    return GRAPH_SUCCESS;
  } else if (CheckNodeCanBeDeleted(node)) {
    GELOGD("node: %s, type: %s does not change memory, just delete", node->GetName().c_str(), node->GetType().c_str());

    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    auto in_anchors = out_anchor->GetPeerInDataAnchors();
    GE_CHK_STATUS_RET(RemoveNode(node, graph),
                      "[Remove][Node] %s from graph:%s failed", node->GetName().c_str(), graph->GetName().c_str());
    GELOGI("remove node: %s, type: %s.", node->GetName().c_str(), node->GetType().c_str());
    for (auto &in_anchor : in_anchors) {
      GE_CHECK_NOTNULL(in_anchor);
      GE_CHK_STATUS_RET(UpdateSrcAttr(in_anchor->GetPeerOutAnchor(), out_anchor, in_anchor),
                        "[Update][SrcAttr] failed");
      GE_CHK_STATUS_RET(RecursiveInDepth(in_anchor, graph),
                        "[Call][RecursiveInDepth] failed, graph:%s", graph->GetName().c_str());
    }
  } else if (trans_op_.empty() || !DescAreSymmetry(trans_op_.top(), node)) {
    GELOGD("node: %s, type: %s can't be offset, push to trans_op_", node->GetName().c_str(), node->GetType().c_str());

    trans_op_.push(node);
    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS_RET(RecursiveInDepth(in_anchor, graph),
                        "[Call][RecursiveInDepth] failed, graph:%s", graph->GetName().c_str());
    }

    if (node->GetOutDataNodesSize() == 0) {
      GE_CHK_STATUS_RET(RemoveNode(node, graph),
                        "[Remove][Node] %s from graph:%s failed", node->GetName().c_str(), graph->GetName().c_str());
      GELOGI("backtracking, trans op: %s, type: %s will be removed", node->GetName().c_str(), node->GetType().c_str());
    }
    GELOGD("backtracking, trans_op_ fall back. pop node: %s, type: %s.", trans_op_.top()->GetName().c_str(),
           trans_op_.top()->GetType().c_str());
    trans_op_.pop();
  } else if (DescAreSymmetry(trans_op_.top(), node)) {
    GELOGD("current node: %s, type: %s can be offset with node: %s, type %s", node->GetName().c_str(),
           node->GetType().c_str(), trans_op_.top()->GetName().c_str(), trans_op_.top()->GetType().c_str());
    GELOGD("offset_op_ push node: %s, type: %s.", trans_op_.top()->GetName().c_str(),
           trans_op_.top()->GetType().c_str());
    offset_op_.push(trans_op_.top());

    auto in_data_anchor = node->GetInDataAnchor(0);
    GE_CHECK_NOTNULL(in_data_anchor);
    auto old_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(old_out_anchor);
    auto new_out_anchor = trans_op_.top()->GetInDataAnchor(0)->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(new_out_anchor);
    GE_IF_BOOL_EXEC(RelinkEdges(new_out_anchor, old_out_anchor, in_data_anchor) != GRAPH_SUCCESS,
                    GELOGE(FAILED, "[Relink][Edges] failed.");
                    return FAILED)
    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    auto in_anchors = out_anchor->GetPeerInDataAnchors();

    GELOGD("begin offset,trans_op_ pop node: %s, type: %s.", trans_op_.top()->GetName().c_str(),
           trans_op_.top()->GetType().c_str());
    GELOGI("the offset node : %s, type: %s will be removed.", node->GetName().c_str(), node->GetType().c_str());
    GE_CHK_STATUS_RET(RemoveNode(node, graph),
                      "[Remove][Node] %s from graph:%s failed", node->GetName().c_str(), graph->GetName().c_str());
    trans_op_.pop();

    for (const auto &in_anchor : in_anchors) {
      GE_CHECK_NOTNULL(in_anchor);
      GE_CHK_STATUS_RET(UpdateSrcAttr(in_anchor->GetPeerOutAnchor(), out_anchor, in_anchor),
                        "[Update][SrcAttr] failed");
      GE_CHK_STATUS_RET(RecursiveInDepth(in_anchor, graph),
                        "[Call][RecursiveInDepth] failed, graph:%s", graph->GetName().c_str());
    }

    GELOGD("backtracking, trans_op_ push node: %s, type: %s.", offset_op_.top()->GetName().c_str(),
           offset_op_.top()->GetType().c_str());
    trans_op_.push(offset_op_.top());
    offset_op_.pop();
  }
  temp_depth--;
  return GRAPH_SUCCESS;
}

bool TransOpDepthFusionPass::CheckNodeCanBeDeleted(const NodePtr &node) {
  bool is_shape_unknown = false;
  if (NodeUtils::GetNodeUnknownShapeStatus(*node, is_shape_unknown) == GRAPH_SUCCESS) {
    if (is_shape_unknown) {
      GELOGI("op:%s is unknown shape, can not be deleted.",
             node->GetName().c_str());
      return false;
    }
  }
  return node->GetType() == RESHAPE || node->GetType() == REFORMAT || node->GetType() == SQUEEZE ||
         node->GetType() == EXPANDDIMS;
}

bool TransOpDepthFusionPass::DescAreSymmetry(const NodePtr &src_node, const NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr || src_node->GetOpDesc() == nullptr ||
      dst_node->GetOpDesc() == nullptr) {
    return false;
  }
  const auto &src_input_desc = src_node->GetOpDesc()->MutableInputDesc(0);
  const auto &dst_output_desc = dst_node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(src_input_desc, return false);
  GE_CHECK_NOTNULL_EXEC(dst_output_desc, return false);
  const auto &src_input_dtype = src_input_desc->GetDataType();
  const auto &src_input_format = src_input_desc->GetFormat();
  const auto &src_input_shape = src_input_desc->GetShape().GetDims();
  const auto &dst_output_dtype = dst_output_desc->GetDataType();
  const auto &dst_output_format = dst_output_desc->GetFormat();
  const auto &dst_output_shape = dst_output_desc->GetShape().GetDims();

  if (src_node->GetType() == CAST && dst_node->GetType() == CAST) {
    return src_input_dtype == dst_output_dtype && src_input_format == dst_output_format;
  } else {
    return src_input_dtype == dst_output_dtype && src_input_shape == dst_output_shape &&
           src_input_format == dst_output_format;
  }
}

// If the relationship was changed, the input and src name will be update
graphStatus TransOpDepthFusionPass::UpdateSrcAttr(const OutDataAnchorPtr &new_out_anchor,
                                                  const OutDataAnchorPtr &ori_out_anchor,
                                                  const InDataAnchorPtr &dst_in_anchor) {
  if (dst_in_anchor == nullptr || dst_in_anchor->GetOwnerNode() == nullptr ||
      dst_in_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
    GELOGW("dst_in_anchor or it's owner node and  op_desc is nullptr");
    return GRAPH_SUCCESS;
  }
  GE_CHECK_NOTNULL(new_out_anchor);
  GE_CHECK_NOTNULL(new_out_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(ori_out_anchor);
  GE_CHECK_NOTNULL(ori_out_anchor->GetOwnerNode());
  auto new_name = new_out_anchor->GetOwnerNode()->GetName();
  auto ori_name = ori_out_anchor->GetOwnerNode()->GetName();
  auto dst_desc = dst_in_anchor->GetOwnerNode()->GetOpDesc();

  auto ori_src_name = dst_desc->GetSrcName();
  auto ori_input_name = dst_desc->GetInputName();

  std::vector<string> new_src_name;
  std::vector<string> new_input_name;

  if (ori_src_name.empty()) {
    new_src_name.push_back(new_name);
  } else {
    for (auto &src_name : ori_src_name) {
      if (src_name == ori_name) {
        new_src_name.push_back(new_name);
      } else {
        new_src_name.push_back(src_name);
      }
    }
  }

  if (ori_input_name.empty()) {
    new_input_name.push_back(new_name);
  } else {
    for (auto &input_name : ori_input_name) {
      if (input_name == ori_name) {
        new_input_name.push_back(new_name);
      } else {
        new_input_name.push_back(input_name);
      }
    }
  }
  dst_desc->SetSrcName(new_src_name);
  dst_desc->SetInputName(new_input_name);
  return GRAPH_SUCCESS;
}

/// Relink the offset trans op with it's former neighbor's father node.
/// Note: control edge will be added to link the two offset ops, if the former op
/// has in control nodes
graphStatus TransOpDepthFusionPass::RelinkEdges(const OutDataAnchorPtr &new_out_anchor,
                                                const OutDataAnchorPtr &old_out_anchor,
                                                const InDataAnchorPtr &in_data_anchor) {
  if (new_out_anchor == nullptr || old_out_anchor == nullptr || in_data_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param anchor info has nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] new_out_anchor or old_out_anchor or in_data_anchor is nullptr");
    return GRAPH_FAILED;
  }
  if (new_out_anchor->GetOwnerNode() == nullptr || old_out_anchor->GetOwnerNode() == nullptr ||
      in_data_anchor->GetOwnerNode() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param anchor info owner node has nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] anchor's owner node has nullptr");
    return GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(old_out_anchor, in_data_anchor),
                    "[Remove][Edge] between %s and %s failed",
                    old_out_anchor->GetOwnerNode()->GetName().c_str(),
                    in_data_anchor->GetOwnerNode()->GetName().c_str());
  GE_CHK_STATUS_RET(GraphUtils::AddEdge(new_out_anchor, in_data_anchor),
                    "[Add][Edge] between %s and %s failed",
                    new_out_anchor->GetOwnerNode()->GetName().c_str(),
                    in_data_anchor->GetOwnerNode()->GetName().c_str());
  GELOGD(
      "relink edges before remove node, remove data edge between node: %s, "
      "type: %s and node: %s, type: %s.",
      old_out_anchor->GetOwnerNode()->GetName().c_str(), old_out_anchor->GetOwnerNode()->GetType().c_str(),
      in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetOwnerNode()->GetType().c_str());
  GELOGD(
      "relink edges before remove node, add data edge between node: %s, "
      "type: %s and node: %s, type: %s.",
      new_out_anchor->GetOwnerNode()->GetName().c_str(), new_out_anchor->GetOwnerNode()->GetType().c_str(),
      in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetOwnerNode()->GetType().c_str());

  bool is_linked = false;
  auto dst_node = in_data_anchor->GetOwnerNode();
  auto src_node = old_out_anchor->GetOwnerNode();
  auto in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!in_ctrl_nodes.empty()) {
    auto iter = std::find(in_ctrl_nodes.begin(), in_ctrl_nodes.end(), src_node);
    is_linked = iter != in_ctrl_nodes.end();
  }
  if (!src_node->GetInControlNodes().empty() && !is_linked) {
    auto out_ctrl_anchor = src_node->GetOutControlAnchor();
    auto in_ctrl_anchor = dst_node->GetInControlAnchor();
    GE_CHK_STATUS_RET(GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor),
                      "[Add][ControlEdge] between %s and %s failed",
                      src_node->GetName().c_str(), dst_node->GetName().c_str());
    GELOGD(
        "relink edges before remove node, add control edge between node: %s,"
        " type: %s and node: %s, type: %s.",
        src_node->GetName().c_str(), src_node->GetType().c_str(), dst_node->GetName().c_str(),
        dst_node->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

// Remove trans op by using interface: IsolateNode & RemoveNodeWithoutRelink
graphStatus TransOpDepthFusionPass::RemoveNode(const NodePtr &node, const ge::ComputeGraphPtr &graph) {
  if (node == nullptr || graph == nullptr) {
    return GRAPH_FAILED;
  }
  if (GraphUtils::IsolateNode(node, {0}) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate node:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] failed, node name:%s, node type:%s",
           node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] without relink failed, node name:%s, node type:%s ",
           node->GetName().c_str(), node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
