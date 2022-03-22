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

#include "graph/passes/merge_input_memcpy_pass.h"

#include "common/ge/ge_util.h"
#include "external/ge/ge_api_types.h"
#include "common/omg_util.h"

namespace ge {
Status MergeInputMemcpyPass::Run(ComputeGraphPtr graph) {
  GELOGD("MergeInputMemcpyPass Enter");
  std::unordered_map<NodePtr, std::vector<NodePtr>> switch_groups;
  for (const auto &node : graph->GetDirectNode()) {
    std::string type;
    GE_CHK_STATUS_RET(GetOriginalType(node, type),
                      "[Get][OriginalType] of node in graph:%s failed.", graph->GetName().c_str());
    if ((type != MERGE) && (type != REFMERGE)) {
      continue;
    }

    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_CHK_STATUS_RET(AddMemcpyAsyncNodes(graph, node, node->GetOpDesc()->HasAttr(ATTR_INSERT_BY_MBATCH)),
                      "[Add][MemcpyAsyncNodes] failed, graph:%s, node:%s.", graph->GetName().c_str(),
                      node->GetName().c_str());
  }

  GELOGD("MergeInputMemcpyPass Leave");
  return SUCCESS;
}

///
/// @brief Add MemcpyAsync Op as Merge in_node
/// @param [in] graph
/// @param [in] node
/// @param [in] multi_batch_flag
/// @return Status
///
Status MergeInputMemcpyPass::AddMemcpyAsyncNodes(const ComputeGraphPtr &graph, const NodePtr &node,
                                                 bool multi_batch_flag) {
  for (const InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    const std::string &type = in_node->GetType();
    // For WhileLoop no need memcpy for merge.
    GE_IF_BOOL_EXEC((type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) || (type == REFNEXTITERATION),
                    continue);

    const std::string &memcpy_name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx());
    NodePtr memcpy_node = CreateMemcpyAsyncNode(graph, memcpy_name, peer_out_anchor, multi_batch_flag);
    GE_CHK_BOOL_EXEC(memcpy_node != nullptr, return FAILED,
                     "[Create][MemcpyAsyncNode] failed, memcpy_name:%s.", memcpy_name.c_str());
    GE_CHK_STATUS(GraphUtils::RemoveEdge(peer_out_anchor, in_data_anchor),
                  "[Remove][Edge] between %s and %s failed.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                  node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(peer_out_anchor, memcpy_node->GetInDataAnchor(0)),
                  "[Add][Edge] between %s and %s failed.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                  memcpy_node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(memcpy_node->GetOutDataAnchor(0), in_data_anchor),
                  "[Add][Edge] between %s and %s failed.", memcpy_node->GetName().c_str(),
                  node->GetName().c_str());
  }

  return SUCCESS;
}

///
/// @brief Add MemcpyAsync Node
/// @param [in] graph
/// @param [in] name
/// @param [in] out_data_anchor
/// @param [in] multi_batch_flag
/// @return ge::NodePtr
///
NodePtr MergeInputMemcpyPass::CreateMemcpyAsyncNode(const ComputeGraphPtr &graph, const std::string &name,
                                                    const OutDataAnchorPtr &out_data_anchor, bool multi_batch_flag) {
  OpDescPtr pre_op_desc = out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_op_desc != nullptr,
                   REPORT_INNER_ERROR("E19999", "opdesc of pre node is nullptr, check invalid");
                   return nullptr, "[Get][OpDesc] failed, OpDesc of pre node is invalid.");

  const std::string &memcpy_type = multi_batch_flag ? MEMCPYADDRASYNC : MEMCPYASYNC;
  const std::string &node_name = name + "_" + memcpy_type;
  GELOGI("Create MemcpyAsync op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, memcpy_type);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "Create OpDesc failed, node_name:%s", node_name.c_str());
    GELOGE(FAILED, "[Create][OpDesc] failed, MemcpyAsync:%s.", node_name.c_str());
    return nullptr;
  }

  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "[Add][InputDesc] to op:%s(%s) failed", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add output to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "[Add][OutputDesc] to op:%s(%s) failed", op_desc->GetName().c_str(), op_desc->GetType().c_str());

  return graph->AddNode(op_desc);
}
}  // namespace ge
