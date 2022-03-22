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
#include "graph/passes/cond_pass.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"

namespace {
  const std::string kStringLength = "StringLength";
}

namespace ge {
Status CondPass::Run(NodePtr &node) {
  ComputeGraphPtr graph = nullptr;
  OutDataAnchorPtr peer_out_anchor = nullptr;
  InDataAnchorPtr cond_in_anchor = nullptr;
  Status ret = GetCondInfo(node, graph, peer_out_anchor, cond_in_anchor);
  if (ret == NOT_CHANGED) {
    return SUCCESS;
  } else if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][CondInfo] for node %s failed.", node->GetName().c_str());
    return FAILED;
  }

  /// cond
  /// 1. NonScalar: cond->Size(int32)->If / NetOutput(while)
  /// 2. String Scalar: cond->StringLength(int32)->If / NetOutput(while)
  /// 3. bool / float / double / uint8 / int16 / int8 / int64 Scalar: cond->Cast(2int32)->If / NetOutput(while)
  /// 4. Int32 Scalar: cond->If / NetOutput(while)
  OpDescPtr op_desc = cond_in_anchor->GetOwnerNode()->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Handle cond for node %s.", op_desc->GetName().c_str());
  GeTensorDesc cond_tensor = op_desc->GetInputDesc(cond_in_anchor->GetIdx());
  if (cond_tensor.MutableShape().GetDim(0) == UNKNOWN_DIM_NUM) {
    GELOGI("Output tensor rank of Cond is unknown.");
    if (cond_tensor.GetDataType() == DT_STRING) {
      GE_CHK_STATUS_RET(HandleStringCond(graph, peer_out_anchor, cond_in_anchor),
                        "[Handle][StringCond] for op:%s failed.", op_desc->GetName().c_str())
    }
    return SUCCESS;
  }
  if (!cond_tensor.GetShape().IsScalar()) {
    GE_CHK_STATUS_RET(HandleNonScalarCond(graph, peer_out_anchor, cond_in_anchor),
                      "[Handle][NonScalarCond] for op:%s failed.", op_desc->GetName().c_str())
  } else {
    switch (cond_tensor.GetDataType()) {
      case DT_STRING:
        GE_CHK_STATUS_RET(HandleStringCond(graph, peer_out_anchor, cond_in_anchor),
                          "[Handle][StringCond] for op:%s failed.", op_desc->GetName().c_str())
        break;
      case DT_BOOL:
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_UINT8:
      case DT_INT16:
      case DT_INT8:
      case DT_INT64:
        GE_CHK_STATUS_RET(HandleScalarCond(graph, peer_out_anchor, cond_in_anchor, cond_tensor.GetDataType()),
                          "[Handle][ScalarCond] for op:%s failed.", op_desc->GetName().c_str())
        break;
      case DT_INT32:
        break;
      default:
        REPORT_INNER_ERROR("E19999",
                           "data_type:%d of index:%d input tensor in op:%s(%s) check invalid",
                           cond_tensor.GetDataType(), cond_in_anchor->GetIdx(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Check][Param] data_type:%d of index:%d input tensor in op:%s(%s) is invalid",
               cond_tensor.GetDataType(), cond_in_anchor->GetIdx(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
    }
  }

  cond_tensor.SetDataType(DT_INT32);
  cond_tensor.SetOriginDataType(DT_INT32);
  cond_tensor.SetShape(GeShape());
  cond_tensor.SetOriginShape(GeShape());
  if (op_desc->UpdateInputDesc(cond_in_anchor->GetIdx(), cond_tensor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Update input desc of op:%s(%s) failed, index:%d",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), cond_in_anchor->GetIdx());
    GELOGE(FAILED, "[Update][InputDesc] for op:%s(%s) failed, index:%d",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), cond_in_anchor->GetIdx());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Get cond info for if / while
/// @param [in] node: If / While op
/// @param [out] graph: owner_graph of if node / while_cond subgraph
/// @param [out] peer_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: cond_input
/// @return Status
///
Status CondPass::GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                             InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  std::string type = node->GetType();
  if (kIfOpTypes.count(type) != 0) {
    if (GetCondInfoForIf(node, graph, peer_out_anchor, cond_in_anchor) != SUCCESS) {
      GELOGE(FAILED, "[Get][CondInfo] for if node:%s failed.", node->GetName().c_str());
      return FAILED;
    }
  } else if (kWhileOpTypes.count(type) != 0) {
    if (GetCondInfoForWhile(node, graph, peer_out_anchor, cond_in_anchor) != SUCCESS) {
      GELOGE(FAILED, "[Get][CondInfo] for while node:%s failed.", node->GetName().c_str());
      return FAILED;
    }
  } else {
    GELOGD("no need cond_pass for node %s.", node->GetName().c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

///
/// @brief Get cond info for if node
/// @param [in] node: If op
/// @param [out] graph: owner_graph of if node
/// @param [out] peer_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: cond_input of if
/// @return Status
///
Status CondPass::GetCondInfoForIf(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                                  InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  cond_in_anchor = node->GetInDataAnchor(IF_COND_INPUT);
  GE_CHECK_NOTNULL(cond_in_anchor);
  peer_out_anchor = cond_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  return SUCCESS;
}

///
/// @brief Get cond info for while node
/// @param [in] node: While op
/// @param [out] graph: while_cond subgraph
/// @param [out] peer_out_anchor: peer_cond_anchor
/// @param [out] cond_in_anchor: input of NetOutput in cond_graph
/// @return Status
///
Status CondPass::GetCondInfoForWhile(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                                     InDataAnchorPtr &cond_in_anchor) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::map<std::string, uint32_t> subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  auto iter = subgraph_names_to_index.find(ATTR_NAME_WHILE_COND);
  if (iter == subgraph_names_to_index.end()) {
    REPORT_INNER_ERROR("E19999", "subgraph name:%s not exist in SubgraphNameIndexes map of op:%s(%s), "
                       "check invalid", ATTR_NAME_WHILE_COND.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "subgraph name:%s not exist in SubgraphNameIndexes map of op:%s(%s)", ATTR_NAME_WHILE_COND.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  std::string cond_graph_instance_name = op_desc->GetSubgraphInstanceName(iter->second);
  graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph())->GetSubgraph(cond_graph_instance_name);
  GE_CHECK_NOTNULL(graph);

  NodePtr net_output_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_node);
  // cond_graph has and only has one output
  uint32_t output_num = net_output_node->GetAllInDataAnchorsSize();
  if (output_num != 1) {
    REPORT_INNER_ERROR("E19999", "Input data anchor num:%u of op:%s(%s) not equal to 1, check invalid",
                       output_num, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] output size of cond_graph is invalid, expect 1 but %u exactly, while_node:%s.",
           output_num, node->GetName().c_str());
    return FAILED;
  }

  cond_in_anchor = net_output_node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(cond_in_anchor);
  peer_out_anchor = cond_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  return SUCCESS;
}

///
/// @brief Process Cond Op with non-scalar cond_input: cond->Size->If / NetOutput(while)
/// @param [in] graph
/// @param [in] peer_out_anchor: peer_cond_anchor
/// @param [in] cond_in_anchor: cond_input
/// @return Status
///
Status CondPass::HandleNonScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                                     const InDataAnchorPtr &cond_in_anchor) {
  GELOGI("Handle cond with non-scalar cond-input.");
  return InsertNode(graph, peer_out_anchor, cond_in_anchor, SIZE);
}

///
/// @brief Process Cond Op with scalar-string cond_input: cond->StringLength(int32)->If / NetOutput(while)
/// @param [in] graph
/// @param [in] peer_out_anchor: peer_cond_anchor
/// @param [in] cond_in_anchor: cond_input
/// @return Status
///
Status CondPass::HandleStringCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                                  const InDataAnchorPtr &cond_in_anchor) {
  GELOGI("Handle cond with scalar-string cond-input.");
  return InsertNode(graph, peer_out_anchor, cond_in_anchor, kStringLength);
}

///
/// @brief Process Cond Op with scalar cond_input: cond->Cast(2int32)->If / NetOutput(while)
/// @param [in] graph
/// @param [in] peer_out_anchor: peer_cond_anchor
/// @param [in] cond_in_anchor: cond_input
/// @param [in] src_type
/// @return Status
///
Status CondPass::HandleScalarCond(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                                  const InDataAnchorPtr &cond_in_anchor, DataType src_type) {
  GE_CHECK_NOTNULL(cond_in_anchor);
  GE_CHECK_NOTNULL(peer_out_anchor);
  GE_CHECK_NOTNULL(peer_out_anchor->GetOwnerNode()->GetOpDesc());
  GELOGI("Handle cond with scalar cond-input.");

  GeTensorDesc tensor = peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer_out_anchor->GetIdx());
  std::string cast_name = cond_in_anchor->GetOwnerNode()->GetName() + "_Cast";
  NodePtr cast_node = AddCastNode(graph, cast_name, tensor, src_type, DT_INT32);
  if (cast_node == nullptr) {
    GELOGE(FAILED, "[Add][CastNode] failed, name:%s.", cast_name.c_str());
    return FAILED;
  }

  if (GraphUtils::InsertNodeAfter(peer_out_anchor, { cond_in_anchor }, cast_node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Insert Cast node %s(%s) between %s(%s)->%s(%s) failed",
                      cast_node->GetName().c_str(), cast_node->GetType().c_str(),
                      peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                      peer_out_anchor->GetOwnerNode()->GetType().c_str(),
                      cond_in_anchor->GetOwnerNode()->GetName().c_str(),
                      cond_in_anchor->GetOwnerNode()->GetType().c_str());
    GELOGE(FAILED, "[Insert][CastNode] %s between %s->%s failed.",
           cast_node->GetName().c_str(), peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           cond_in_anchor->GetOwnerNode()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Insert node
/// @param [in] graph
/// @param [in] peer_out_anchor
/// @param [in] in_data_anchor
/// @param [in] type
/// @return Status
///
Status CondPass::InsertNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &peer_out_anchor,
                            const InDataAnchorPtr &in_data_anchor, const std::string &type) {
  GE_CHECK_NOTNULL(peer_out_anchor);
  GE_CHECK_NOTNULL(in_data_anchor);
  GELOGD("Begin to insert %s node.", type.c_str());

  GE_CHECK_NOTNULL(peer_out_anchor->GetOwnerNode()->GetOpDesc());
  GE_CHECK_NOTNULL(in_data_anchor->GetOwnerNode()->GetOpDesc());
  GeTensorDesc in_tensor = peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer_out_anchor->GetIdx());
  GeTensorDesc out_tensor = in_data_anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx());
  out_tensor.SetDataType(DT_INT32);
  out_tensor.SetOriginDataType(DT_INT32);
  out_tensor.SetShape(in_tensor.GetShape());
  out_tensor.SetOriginShape(in_tensor.GetOriginShape());

  OpDescBuilder op_desc_builder(in_data_anchor->GetOwnerNode()->GetName() + "_" + type, type);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_tensor).AddOutput("y", out_tensor).Build();
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "Create op_desc:%s(%s) failed",
                      (in_data_anchor->GetOwnerNode()->GetName() + "_" + type).c_str(), type.c_str());
    GELOGE(FAILED, "[Create][OpDesc] %s(%s) failed.",
           (in_data_anchor->GetOwnerNode()->GetName() + "_" + type).c_str(), type.c_str());
    return FAILED;
  }
  NodePtr new_node = graph->AddNode(op_desc);
  if (new_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  AddRePassNode(new_node);

  if (GraphUtils::InsertNodeAfter(peer_out_anchor, { in_data_anchor }, new_node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Insert node %s(%s) between %s(%s)->%s(%s) failed",
                      new_node->GetName().c_str(), new_node->GetType().c_str(),
                      peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                      peer_out_anchor->GetOwnerNode()->GetType().c_str(),
                      in_data_anchor->GetOwnerNode()->GetName().c_str(),
                      in_data_anchor->GetOwnerNode()->GetType().c_str());
    GELOGE(FAILED, "[Insert][Node] %s(%s) between %s(%s)->%s(%s) failed",
           new_node->GetName().c_str(), new_node->GetType().c_str(),
           peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetOwnerNode()->GetType().c_str(),
           in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetOwnerNode()->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Add cast node
/// @param [in] graph
/// @param [in] name
/// @param [in] tensor
/// @param [in] src
/// @param [in] dst
/// @return NodePtr
///
NodePtr CondPass::AddCastNode(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &tensor,
                              DataType src, DataType dst) {
  GELOGI("Begin to create cast op: %s, from %d to %d", name.c_str(), src, dst);

  GeTensorDesc in_tensor = tensor;
  in_tensor.SetDataType(src);
  in_tensor.SetOriginDataType(src);
  GeTensorDesc out_tensor = tensor;
  out_tensor.SetDataType(dst);
  out_tensor.SetOriginDataType(dst);
  OpDescBuilder op_desc_builder(name, CAST);
  OpDescPtr cast_desc = op_desc_builder.AddInput("x", in_tensor).AddOutput("y", out_tensor).Build();
  if (cast_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "Create op_desc:%s(%s) failed", name.c_str(), CAST);
    GELOGE(FAILED, "[Create][OpDesc] failed, name:%s(%s).", name.c_str(), CAST);
    return nullptr;
  }
  if (!(AttrUtils::SetInt(cast_desc, CAST_ATTR_SRCT, src) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DSTT, dst) &&
        AttrUtils::SetInt(cast_desc, CAST_ATTR_DST_TYPE, dst) &&
        AttrUtils::SetBool(cast_desc, CAST_ATTR_TRUNCATE, false))) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s, %s, %s, %s to node:%s(%s) not all success",
                      CAST_ATTR_SRCT.c_str(), CAST_ATTR_DSTT.c_str(),
                      CAST_ATTR_DST_TYPE.c_str(), CAST_ATTR_TRUNCATE.c_str(),
                      cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s, %s, %s, %s to node:%s(%s) not all success",
           CAST_ATTR_SRCT.c_str(), CAST_ATTR_DSTT.c_str(),
           CAST_ATTR_DST_TYPE.c_str(), CAST_ATTR_TRUNCATE.c_str(),
           cast_desc->GetName().c_str(), cast_desc->GetType().c_str());
    return nullptr;
  }

  NodePtr cast_node = graph->AddNode(cast_desc);
  if (cast_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      cast_desc->GetName().c_str(), cast_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           cast_desc->GetName().c_str(), cast_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }
  AddRePassNode(cast_node);

  return cast_node;
}
}  // namespace ge
