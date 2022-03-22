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

#include "graph/passes/replace_transshape_pass.h"

#include <string>

#include "common/ge/ge_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "common/omg_util.h"
#include "graph/utils/graph_utils.h"

namespace ge {
Status ReplaceTransShapePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == TRANSSHAPE) {
      auto ret = ReplaceTransShapeNode(graph, node);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "Trans shape node %s failed", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ReplaceTransShapePass::ReplaceTransShapeNode(ComputeGraphPtr &graph, NodePtr &trans_shape_node) {
  std::string op_type;
  auto ret = GetOriginalType(trans_shape_node, op_type);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get OriginalType of op:%s(%s) failed",
                      trans_shape_node->GetName().c_str(), trans_shape_node->GetType().c_str());
    GELOGE(FAILED, "[Get][OriginalType] of op:%s(%s) failed",
           trans_shape_node->GetName().c_str(), trans_shape_node->GetType().c_str());
    return FAILED;
  }
  auto src_op_desc = trans_shape_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_op_desc);

  std::string node_name = trans_shape_node->GetName() + "ToMemcpy";
  auto dst_op_desc = MakeShared<OpDesc>(node_name, MEMCPYASYNC);
  if (dst_op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return FAILED;
  }
  GELOGI("Create memcpy Op, name=%s.", node_name.c_str());
  for (InDataAnchorPtr &in_anchor : trans_shape_node->GetAllInDataAnchors()) {
    auto ret = dst_op_desc->AddInputDesc(src_op_desc->GetInputDesc(in_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                        dst_op_desc->GetName().c_str(), dst_op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed",
             dst_op_desc->GetName().c_str(), dst_op_desc->GetType().c_str());
      return FAILED;
    }
  }
  for (OutDataAnchorPtr &out_anchor : trans_shape_node->GetAllOutDataAnchors()) {
    auto ret = dst_op_desc->AddOutputDesc(src_op_desc->GetOutputDesc(out_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                        src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][OutputDesc] to op:%s(%s) failed",
             src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str());
      return FAILED;
    }
  }
  NodePtr memcpy_node = graph->AddNode(dst_op_desc);
  GE_CHECK_NOTNULL(memcpy_node);

  for (InDataAnchorPtr &in_data_anchor : trans_shape_node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor),
                  "[Remove][Edge] between %s and %s failed.",
                  peer_out_anchor->GetOwnerNode()->GetName().c_str(), trans_shape_node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, memcpy_node->GetInDataAnchor(in_data_anchor->GetIdx())),
                  "[Add][Edge] between %s and %s failed.",
                  peer_out_anchor->GetOwnerNode()->GetName().c_str(), memcpy_node->GetName().c_str());
  }

  for (OutDataAnchorPtr &out_data_anchor : trans_shape_node->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor),
                    "[Remove][Edge] between %s and %s failed.",
                    trans_shape_node->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
      GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(out_data_anchor->GetIdx()), peer_in_anchor),
                    "[Add][Edge] between %s and %s failed.",
                    memcpy_node->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
    }
  }
  ReplaceControlEdges(trans_shape_node, memcpy_node);
  return SUCCESS;
}

void ReplaceTransShapePass::CopyControlEdges(NodePtr &old_node, NodePtr &new_node, bool input_check_flag) {
  GE_CHECK_NOTNULL_JUST_RETURN(old_node);
  GE_CHECK_NOTNULL_JUST_RETURN(new_node);
  GE_IF_BOOL_EXEC(old_node == new_node, return);
  for (NodePtr &node : old_node->GetInControlNodes()) {
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(!out_control_anchor->IsLinkedWith(new_node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(out_control_anchor, new_node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    node->GetName().c_str(), new_node->GetName().c_str());
    });
  }

  for (NodePtr &node : old_node->GetOutControlNodes()) {
    GE_IF_BOOL_EXEC(!new_node->GetOutControlAnchor()->IsLinkedWith(node->GetInControlAnchor()), {
      GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                    "[Add][ControlEdge] between %s and %s failed.",
                    new_node->GetName().c_str(), node->GetName().c_str());
    });
  }
}

void ReplaceTransShapePass::RemoveControlEdges(NodePtr &node) {
  GE_CHECK_NOTNULL_JUST_RETURN(node);
  for (NodePtr &in_node : node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  in_node->GetName().c_str(), node->GetName().c_str());
  }

  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (auto &in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_STATUS(GraphUtils::RemoveEdge(out_data_anchor, in_ctrl_anchor),
                    "[Remove][Edge] between %s and %s failed.",
                    node->GetName().c_str(), in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    }
  }

  auto out_control_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL_JUST_RETURN(out_control_anchor);
  for (auto &peer_anchor : out_control_anchor->GetPeerAnchors()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(out_control_anchor, peer_anchor),
                  "[Remove][OutCtlEdge] between %s and %s failed.",
                  node->GetName().c_str(), peer_anchor->GetOwnerNode()->GetName().c_str());
  }
}

void ReplaceTransShapePass::ReplaceControlEdges(NodePtr &old_node, NodePtr &new_node) {
  GE_IF_BOOL_EXEC(old_node == new_node, return);
  CopyControlEdges(old_node, new_node);
  RemoveControlEdges(old_node);
}
}
