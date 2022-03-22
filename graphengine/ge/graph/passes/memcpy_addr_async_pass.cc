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

#include "graph/passes/memcpy_addr_async_pass.h"

#include "common/ge/ge_util.h"
#include "framework/common/debug/log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
Status MemcpyAddrAsyncPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetGraphUnknownFlag()) {
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == STREAMSWITCH) {
        auto sub_graph = node->GetOwnerComputeGraph();
        if (sub_graph != nullptr && !sub_graph->GetGraphUnknownFlag()) {
          GE_CHK_STATUS_RET(AddMemcpyAsyncNode(node),
                            "[Add][MemcpyAsyncNode] for node:%s in known subgraph:%s failed.",
                            node->GetName().c_str(), sub_graph->GetName().c_str());
        }
      }
    }
    GELOGD("Graph[%s] is unknown graph, skip.", graph->GetName().c_str());
    return SUCCESS;
  }

  int64_t value = 0;
  rtError_t rt_ret = rtGetRtCapability(FEATURE_TYPE_MEMCPY, MEMCPY_INFO_SUPPORT_ZEROCOPY, &value);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtGetRtCapability failed, ret = 0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtGetRtCapability] failed, ret = 0x%x.", rt_ret);
    return RT_FAILED;
  }

  if (value == RT_CAPABILITY_NOT_SUPPORT) {
    GELOGW("Not support zero copy, skip it.");
    return SUCCESS;
  }

  for (auto &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);

    if (op_desc->GetType() == STREAMSWITCHN || op_desc->GetType() == STREAMMERGE) {
      Status ret = AddMemcpyAddrAsyncNode(graph, node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Add][MemcpyAddrAsyncNode] for %s in graph:%s failed.", node->GetName().c_str(),
               graph->GetName().c_str());
        return ret;
      }
    }
    // handle data->netoutput, const->netoutput in root graph, use mem_addr_async to improve performance
    if (op_desc->GetType() == NETOUTPUT) {
      // check this netoutput is on root graph
      if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        Status ret = InsertMemAddrAsyncNodeBeforeNetoutput(node->GetOwnerComputeGraph(), node);
        if (ret != SUCCESS) {
          GELOGE(ret, "[Insert][MemAddrAsyncNode] Before Netoutput for node:%s in graph:%s failed.",
                 node->GetName().c_str(), graph->GetName().c_str());
          return ret;
        }
      }
    }
  }
  return SUCCESS;
}

Status MemcpyAddrAsyncPass::AddMemcpyAsyncNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GELOGI("Start add memcpyasync node in front of node %s", node->GetName().c_str());
  known_sub_graph_ = true;
  auto sub_graph = node->GetOwnerComputeGraph();
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    auto memcpy_async_node = CreateMemcpyAddrAsyncNode(sub_graph, peer_out_anchor, node);
    if (memcpy_async_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Create][MemcpyAddrAsyncNode] for node:%s in subgraph failed.",
             node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    Status ret = InsertMemcpyAddrAsyncNode(peer_out_anchor, in_data_anchor, memcpy_async_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Insert][MemcpyAddrAsyncNode] failed, memcpy_async_node:%s.", memcpy_async_node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status MemcpyAddrAsyncPass::AddMemcpyAddrAsyncNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGI("Start AddMemcpyAddrAsyncNode for %s.", node->GetName().c_str());
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();

    if (in_node->GetType() == DATA) {
      ComputeGraphPtr owner_graph = in_node->GetOwnerComputeGraph();
      GE_CHECK_NOTNULL(owner_graph);
      // Data is in parent_graph
      if (owner_graph->GetParentGraph() == nullptr) {
        GELOGI("Need to insert MemcpyAddrAsync directly when data in parent graph.");
        NodePtr memcpy_addr_async_node = CreateMemcpyAddrAsyncNode(graph, peer_out_anchor, node);
        GE_IF_BOOL_EXEC(memcpy_addr_async_node == nullptr,
                        GELOGE(INTERNAL_ERROR, "[Create][MemcpyAddrAsyncNode] failed, node:%s.",
                               node->GetName().c_str());
                        return INTERNAL_ERROR);

        Status ret = InsertMemcpyAddrAsyncNode(peer_out_anchor, in_data_anchor, memcpy_addr_async_node);
        GE_IF_BOOL_EXEC(ret != SUCCESS,
                        GELOGE(ret, "[Insert][MemcpyAddrAsyncNode] failed, memcpy_addr_async_node:%s.",
                               memcpy_addr_async_node->GetName().c_str());
                        return ret);
      } else {
        uint32_t parent_index = 0;
        if (!AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
          REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                            in_node->GetName().c_str(), in_node->GetType().c_str());
          GELOGE(INTERNAL_ERROR, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                 in_node->GetName().c_str(), in_node->GetType().c_str());
          return INTERNAL_ERROR;
        }
        // Data is in sub_graph
        GELOGI("Need to find data in parent graph, then insert MemcpyAddrAsync.");
        NodePtr parent_node = owner_graph->GetParentNode();
        user_data_for_known_ = in_node;
        out_of_user_data_for_known_ = node;
        peer_out_anchor_for_known_ = peer_out_anchor;
        in_anchor_for_known_ = in_data_anchor;
        FindUserData(parent_node, parent_index);
        if (find_user_data_) {
          GELOGI("Insert memcpy_addr_async for non_dynamic.");
          GE_CHECK_NOTNULL(peer_out_anchor_);
          NodePtr memcpy_addr_async_node = CreateMemcpyAddrAsyncNode(graph, peer_out_anchor_, out_of_user_data_);
          GE_IF_BOOL_EXEC(memcpy_addr_async_node == nullptr,
                          GELOGE(INTERNAL_ERROR, "[Create][MemcpyAddrAsyncNode] failed, out_of_user_data_:%s.",
                                 out_of_user_data_->GetName().c_str());
                          return INTERNAL_ERROR);

          Status ret = InsertMemcpyAddrAsyncNode(peer_out_anchor_, in_anchor_, memcpy_addr_async_node);
          GE_IF_BOOL_EXEC(ret != SUCCESS,
                          GELOGE(ret, "[Insert][MemcpyAddrAsyncNode] failed, memcpy_addr_async_node:%s.",
                                 memcpy_addr_async_node->GetName().c_str());
                          return ret);
        }
        if (find_user_data_for_known_) {
          GELOGI("Insert memcpy_addr_async for known graph.");
          auto sub_graph = user_data_for_known_->GetOwnerComputeGraph();
          NodePtr memcpy_addr_async_node =
              CreateMemcpyAddrAsyncNode(sub_graph, peer_out_anchor_for_known_, out_of_user_data_for_known_);
          GE_IF_BOOL_EXEC(memcpy_addr_async_node == nullptr,
                          GELOGE(INTERNAL_ERROR,
                                 "[Create][MemcpyAddrAsyncNode] for known failed, out_of_user_data_for_known_:%s",
                                 out_of_user_data_for_known_->GetName().c_str());
                          return INTERNAL_ERROR);

          Status ret =
              InsertMemcpyAddrAsyncNode(peer_out_anchor_for_known_, in_anchor_for_known_, memcpy_addr_async_node);
          GE_IF_BOOL_EXEC(ret != SUCCESS,
                          GELOGE(ret, "[Insert][MemcpyAddrAsyncNode] for known failed, memcpy_addr_async_node:%s.",
                                 memcpy_addr_async_node->GetName().c_str());
                          return ret);
        }
      }
    }
  }
  return SUCCESS;
}

void MemcpyAddrAsyncPass::FindUserDataForKnown(const NodePtr &parent_node, uint32_t &parent_index) {
  GELOGI("Start FindUserDataForKnown of %s.", parent_node->GetName().c_str());
  if (user_data_for_known_->GetOpDesc() == nullptr) {
    GELOGI("Cannot get op_desc of %s.", user_data_for_known_->GetName().c_str());
    return;
  }
  string src_var_name;
  if (ge::AttrUtils::GetStr(user_data_for_known_->GetOpDesc(), REF_VAR_SRC_VAR_NAME, src_var_name)) {
    GELOGI("The data in known graph is variable, no need to insert memcpy_addr_async.");
    find_user_data_for_known_ = false;
    return;
  } else {
    find_user_data_for_known_ = true;
  }
}

void MemcpyAddrAsyncPass::FindUserDataForNonDynamic(const ge::NodePtr &parent_node, uint32_t &parent_index) {
  GELOGI("Start to FindUserDataForNonDynamic of %s.", parent_node->GetName().c_str());
  InDataAnchorPtr in_data_anchor = parent_node->GetInDataAnchor(parent_index);
  OutDataAnchorPtr out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(out_anchor == nullptr,
                  REPORT_INNER_ERROR("E19999", "Index:%u in data node of op:%s(%s) not exist, check invalid",
                                     parent_index, parent_node->GetName().c_str(), parent_node->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Get][PeerOutAnchor] Index:%u in data node of op:%s(%s) not exist",
                         parent_index, parent_node->GetName().c_str(), parent_node->GetType().c_str());
                  return);
  NodePtr in_node = out_anchor->GetOwnerNode();
  GELOGI("in_node of parent_node is %s.", in_node->GetName().c_str());
  if (in_node->GetType() == DATA) {
    if (in_node->GetOwnerComputeGraph()->GetParentGraph() != nullptr) {
      // DATA is in sub graph again, update user_data of known firstly
      user_data_for_known_ = in_node;
      out_of_user_data_for_known_ = parent_node;
      peer_out_anchor_for_known_ = out_anchor;
      in_anchor_for_known_ = in_data_anchor;
      NodePtr pre_in_node = in_node->GetOwnerComputeGraph()->GetParentNode();
      if (!AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                          in_node->GetName().c_str(), in_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
               in_node->GetName().c_str(), in_node->GetType().c_str());
        return;
      }
      FindUserData(pre_in_node, parent_index);
    } else {
      // DATA is in parent graph and not has input
      user_data_ = in_node;
      out_of_user_data_ = parent_node;
      peer_out_anchor_ = out_anchor;
      in_anchor_ = in_data_anchor;
      find_user_data_ = true;
      GELOGI("%s connect with %s, will insert memcpyaddr.", user_data_->GetName().c_str(),
             out_of_user_data_->GetName().c_str());
    }
  } else if (in_node->GetType() == IF || in_node->GetType() == WHILE || in_node->GetType() == CASE) {
    if (!AttrUtils::GetInt(parent_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                        parent_node->GetName().c_str(), parent_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
             parent_node->GetName().c_str(), parent_node->GetType().c_str());
      return;
    }
    FindUserData(in_node, parent_index);
  } else {
    GELOGI("%s connect with %s, which is not user_data.", parent_node->GetName().c_str(), in_node->GetName().c_str());
    find_user_data_ = false;
  }
}

void MemcpyAddrAsyncPass::FindUserData(const NodePtr &parent_node, uint32_t &parent_index) {
  auto parent_op_desc = parent_node->GetOpDesc();
  if (parent_op_desc == nullptr) {
    GELOGI("Cannot get op_desc of %s.", parent_node->GetName().c_str());
    return;
  }
  bool is_unknown_shape = false;
  if (parent_node->GetType() == PARTITIONEDCALL &&
      AttrUtils::GetBool(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape) && !is_unknown_shape) {
    FindUserDataForKnown(parent_node, parent_index);
  } else {
    FindUserDataForNonDynamic(parent_node, parent_index);
  }
}

NodePtr MemcpyAddrAsyncPass::CreateMemcpyAddrAsyncNode(const ComputeGraphPtr &graph,
                                                       const OutDataAnchorPtr &out_data_anchor,
                                                       const NodePtr &out_of_user_data) {
  GELOGD("Start CreateMemcpyAddrAsyncNode.");
  static uint32_t new_node_index = 0;
  OpDescPtr pre_op_desc = out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_op_desc != nullptr,
                   REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
		               return nullptr, "[Get][OpDesc] failed, Op_desc of pre node is invalid.");

  OpDescPtr op_desc = nullptr;
  if (known_sub_graph_) {  // insert memcpyasync node when known sub graph
    string node_name = pre_op_desc->GetName() + "_" + MEMCPYASYNC + "_" + std::to_string(new_node_index++);
    op_desc = MakeShared<OpDesc>(node_name, MEMCPYASYNC);
  } else {
    string node_name = pre_op_desc->GetName() + "_" + MEMCPYADDRASYNC + "_" + std::to_string(new_node_index++);
    op_desc = MakeShared<OpDesc>(node_name, MEMCPYADDRASYNC);
  }
  GE_CHECK_NOTNULL_EXEC(op_desc, REPORT_CALL_ERROR("E19999", "New OpDesc failed"); return nullptr);

  if (op_desc->AddInputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                      pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
           pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str());
    return nullptr;
  }

  if (op_desc->AddOutputDesc(pre_op_desc->GetOutputDesc(out_data_anchor->GetIdx())) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                      pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed",
           pre_op_desc->GetName().c_str(), pre_op_desc->GetType().c_str());
    return nullptr;
  }

  string stream_label;
  if (AttrUtils::GetStr(out_of_user_data->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label);
    GELOGD("Node %s set stream label: %s", op_desc->GetName().c_str(), stream_label.c_str());
  }

  bool rts_label_node = false;
  if (AttrUtils::GetBool(out_of_user_data->GetOpDesc(), ATTR_NAME_RTS_LABEL_NODE, rts_label_node)) {
    (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, rts_label_node);
    GELOGD("Node %s set rts label node attribute", op_desc->GetName().c_str());
  }

  bool labeled_input = false;
  (void)ge::AttrUtils::GetBool(out_of_user_data->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, labeled_input);
  if (labeled_input) {
    if (!ge::AttrUtils::SetBool(out_of_user_data->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, false)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
                        out_of_user_data->GetName().c_str(), out_of_user_data->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
             out_of_user_data->GetName().c_str(), out_of_user_data->GetType().c_str());
      return nullptr;
    }
    if (!ge::AttrUtils::SetBool(op_desc, ATTR_NAME_NODE_CONNECT_INPUT, true)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }

  NodePtr memcpy_addr_async_node = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(memcpy_addr_async_node,
                        REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                          op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                                          graph->GetName().c_str());
                        return nullptr);

  return memcpy_addr_async_node;
}

Status MemcpyAddrAsyncPass::InsertMemcpyAddrAsyncNode(const OutDataAnchorPtr &out_anchor,
                                                      const InDataAnchorPtr &in_anchor, const NodePtr &node) {
  // insert memcpy_addr of each user_data and out_of_user_data
  if (GraphUtils::RemoveEdge(out_anchor, in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
                      out_anchor->GetIdx(), in_anchor->GetOwnerNode()->GetName().c_str(),
                      in_anchor->GetOwnerNode()->GetType().c_str(), in_anchor->GetIdx());
    GELOGE(INTERNAL_ERROR, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
           out_anchor->GetIdx(), in_anchor->GetOwnerNode()->GetName().c_str(),
           in_anchor->GetOwnerNode()->GetType().c_str(), in_anchor->GetIdx());
    return INTERNAL_ERROR;
  }
  if (GraphUtils::AddEdge(out_anchor, node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                      out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
                      out_anchor->GetIdx(), node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
           out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str(),
           out_anchor->GetIdx(), node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (GraphUtils::AddEdge(node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                      node->GetName().c_str(), node->GetType().c_str(), in_anchor->GetOwnerNode()->GetName().c_str(),
                      in_anchor->GetOwnerNode()->GetType().c_str(), in_anchor->GetIdx());
    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
           node->GetName().c_str(), node->GetType().c_str(), in_anchor->GetOwnerNode()->GetName().c_str(),
           in_anchor->GetOwnerNode()->GetType().c_str(), in_anchor->GetIdx());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status MemcpyAddrAsyncPass::InsertMemAddrAsyncNodeBeforeNetoutput(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGD("Start AddMemcpyAddrAsyncNode for %s.", node->GetName().c_str());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto in_node = NodeUtils::GetInDataNodeByIndex(*node, in_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(in_node);
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if ((in_node->GetType() != CONSTANT) &&
       (in_node->GetType() != CONSTANTOP) &&
       (in_node->GetType() != DATA)) {
      continue;
    }
    auto desc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(desc);
    if (IsEmptyTenor(desc->GetOutputDesc(peer_out_anchor->GetIdx()).GetShape())) {
      continue;
    }
    GELOGI("Need to insert MemcpyAddrAsync before netoutput on parent graph.");
    NodePtr memcpy_addr_async_node = CreateMemcpyAddrAsyncNode(graph, peer_out_anchor, in_node);
    GE_IF_BOOL_EXEC(memcpy_addr_async_node == nullptr,
                    GELOGE(INTERNAL_ERROR, "[Create][MemcpyAddrAsyncNode] failed, in_node:%s.",
                           in_node->GetName().c_str());
                    return INTERNAL_ERROR);

    Status ret = InsertMemcpyAddrAsyncNode(peer_out_anchor, in_data_anchor, memcpy_addr_async_node);
    GE_IF_BOOL_EXEC(ret != SUCCESS,
                    GELOGE(ret, "[Insert][MemcpyAddrAsyncNode] failed, memcpy_addr_async_node:%s.",
                           memcpy_addr_async_node->GetName().c_str());
                    return ret);
    GELOGI("Insert mem_addr_async node %s success between %s and %s.", memcpy_addr_async_node->GetName().c_str(),
           in_node->GetName().c_str(), node->GetName().c_str());
    // if src node is const, need to update attr and offset here because this pass process is after offset set.
    if ((in_node->GetType() == CONSTANT) || (in_node->GetType() == CONSTANTOP)) {
      NodeUtils::UpdateIsInputConst(memcpy_addr_async_node);
      auto output_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(output_desc);
      auto output_tensor_desc = output_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
      int64_t data_offset = 0;
      (void)TensorUtils::GetDataOffset(*output_tensor_desc, data_offset);
      auto input_tensor = memcpy_addr_async_node->GetOpDesc()->MutableInputDesc(0);
      GELOGI("Need update const Offset %ld to op [%s]", data_offset, memcpy_addr_async_node->GetName().c_str());
      TensorUtils::SetDataOffset(*input_tensor, data_offset);
      TensorUtils::SetDataOffset(*output_tensor_desc, 0);
    }
  }
  NodeUtils::UpdateIsInputConst(node);
  return SUCCESS;
}

bool MemcpyAddrAsyncPass::IsEmptyTenor(const GeShape &shape) const {
  for (const auto dim : shape.GetDims()) {
    if (dim == 0) {
      return true;
    }
  }
  return false;
}
}  // namespace ge
