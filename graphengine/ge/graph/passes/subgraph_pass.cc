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

#include "graph/passes/subgraph_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
/**
 * @ingroup ge
 * @brief Subgraph optimizer.
 * @param [in] graph: Input ComputeGraph
 * @return: 0 for success / others for fail
 */
Status SubgraphPass::Run(ComputeGraphPtr graph) {
  const bool is_sub_graph = graph->GetParentNode() != nullptr;
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (is_sub_graph && (node->GetType() == DATA)) {
      if (SubgraphInputNode(graph, node) != SUCCESS) {
        GELOGE(FAILED, "[Handle][Input] %s of subgraph:%s failed.",
               node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      continue;
    }

    // NetOutput in subgraph
    if (is_sub_graph && (node->GetType() == NETOUTPUT)) {
      if (SubgraphOutputNode(graph, node) != SUCCESS) {
        GELOGE(FAILED, "[Handle][Output] %s of subgraph:%s failed.",
               node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      continue;
    }

    if (kWhileOpTypes.count(node->GetType()) > 0) {
      // Input->While and Input link to other nodes
      if (WhileInputNodes(graph, node) != SUCCESS) {
        GELOGE(FAILED, "[Handle][Input] of while_body failed, while:%s, graph:%s.",
               node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      // body subgraph of While op
      if (WhileBodySubgraph(graph, node) != SUCCESS) {
        GELOGE(FAILED, "[Handle][WhileBody] failed, while:%s, graph:%s.",
               node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      continue;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check Subgraph Input node
 * @param [in] graph: ComputeGraph.
 * @param [in] node: Data node in Subgraph.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::SubgraphInputNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGD("Handle input_node %s for graph %s.", node->GetName().c_str(), graph->GetName().c_str());
  // Data has and only has one output
  bool input_continues_required_flag = false;
  OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(0);
  std::vector<InDataAnchorPtr> in_anchors;
  for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    input_continues_required_flag =
        input_continues_required_flag || IsInputContinuesRequired(peer_in_anchor->GetOwnerNode());
    in_anchors.emplace_back(peer_in_anchor);
  }
  // Data->InputContinuesRequiredOp in subgraph need memcpy.
  if (input_continues_required_flag) {
    GELOGD("Data %s output_node required continues input.", node->GetName().c_str());
    std::string name = node->GetName() + "_output_0_Memcpy";
    if (InsertMemcpyNode(graph, out_data_anchor, in_anchors, name) != SUCCESS) {
      GELOGE(FAILED, "[Insert][Memcpy] after %s failed.", node->GetName().c_str());
      return FAILED;
    }
  }

  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  // Subgraph Data Node, check for constant input.
  std::string const_type;
  if (!NodeUtils::GetConstOpType(node, const_type)) {
    return SUCCESS;
  }

  const NodePtr &parent_node = graph->GetParentNode();
  if (kWhileOpTypes.count(parent_node->GetType()) != 0) {
    // Constant input to While need memcpy.
    const ComputeGraphPtr &parent_graph = parent_node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(parent_graph);
    const InDataAnchorPtr &in_data_anchor = parent_node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(in_data_anchor);
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    GELOGD("Constant input %s links to While %s.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
           parent_node->GetName().c_str());
    std::string name = parent_node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx()) + "_Memcpy";
    if (InsertMemcpyNode(parent_graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
      GELOGE(FAILED, "[Insert][Memcpy] between %s and %s failed.", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             parent_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check Subgraph Output node
 * @param [in] graph: ComputeGraph.
 * @param [in] node: NetOutput node in Subgraph.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::SubgraphOutputNode(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_node);

    // Need insert memcpy
    //   1. Const->NetOutput in subgraph & parent graph is known
    //   2. AtomicOp->NetOutput in subgraph
    //   3. OutputContinuesRequiredOp->NetOutput in subgraph
    //   4. Data->NetOutput in subgraph but parent_node is not while
    //   5. While->NetOutput in known subgraph
    std::string op_type;
    bool insert_flag =
        (NodeUtils::GetConstOpType(in_node, op_type) && !graph->GetParentGraph()->GetGraphUnknownFlag()) ||
        IsAtomicRequired(in_node, peer_out_anchor->GetIdx()) || IsOutputContinuesRequired(in_node) ||
        ((in_node->GetType() == DATA) && (kWhileOpTypes.count(graph->GetParentNode()->GetType()) == 0)) ||
        (!graph->GetGraphUnknownFlag() && NodeUtils::IsDynamicShape(node) &&
        (kWhileOpTypes.count(in_node->GetType()) != 0));
    if (insert_flag) {
      GELOGD("Insert MemcpyAsync node between %s and %s.", in_node->GetName().c_str(), node->GetName().c_str());
      std::string name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx()) + "_Memcpy";
      if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
        GELOGE(FAILED, "[Insert][Memcpy] between %s and %s failed.",
               in_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check is Input->While and Input link to other nodes
 * @param [in] graph: ComputeGraph.
 * @param [in] node: While node.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::WhileInputNodes(const ComputeGraphPtr &graph, const NodePtr &node) {
  for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(in_node);
    if (in_node->GetType() == VARIABLE || in_node->GetType() == VARHANDLEOP || in_node->GetType() == VARIABLEV2) {
      continue;
    }
    // Input->While and Input link to other nodes need insert memcpy
    if (peer_out_anchor->GetPeerInDataAnchors().size() > 1) {
      GELOGD("Input %s of While %s links to other nodes.", in_node->GetName().c_str(), node->GetName().c_str());
      std::string name = node->GetName() + "_input_" + std::to_string(in_data_anchor->GetIdx()) + "_Memcpy";
      if (InsertMemcpyNode(graph, peer_out_anchor, {in_data_anchor}, name) != SUCCESS) {
        GELOGE(FAILED, "[Insert][Memcpy] between %s and %s failed.",
               in_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check body subgraph of While op
 * @param [in] graph: ComputeGraph.
 * @param [in] node: While node.
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::WhileBodySubgraph(const ComputeGraphPtr &graph, const NodePtr &node) {
  // index of body_subgraph is 1
  ComputeGraphPtr while_body = NodeUtils::GetSubgraph(*node, 1);
  if (while_body == nullptr) {
    REPORT_INNER_ERROR("E19999", "While_body of node:%s(%s) is nullptr, check invalid",
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Get][Subgraph] failed, while_body of %s is nullptr.", node->GetName().c_str());
    return FAILED;
  }
  if (GraphUtils::IsUnknownShapeGraph(while_body)) {
    GELOGI("Unknown shape while_body graph %s no need to insert memcpy.", while_body->GetName().c_str());
    return SUCCESS;
  }

  // insert identity between data and labelswitch in while cond subgraph
  if (NodeUtils::IsDynamicShape(node)) {
    ComputeGraphPtr while_cond = NodeUtils::GetSubgraph(*node, 0);
    GE_CHECK_NOTNULL(while_cond);
    std::vector<NodePtr> cond_data_nodes;
    for (const auto &n : while_cond->GetDirectNode()) {
      if (n->GetType() == DATA) {
        cond_data_nodes.emplace_back(n);
      }
    }
    GE_CHK_STATUS_RET(InsertInputMemcpy(while_cond, cond_data_nodes),
                      "[Insert][InputMemcpy] %s failed.", while_cond->GetName().c_str());
  }

  std::vector<NodePtr> data_nodes;
  std::set<uint32_t> bypass_index;
  NodePtr output_node = nullptr;
  for (const auto &n : while_body->GetDirectNode()) {
    const std::string &type = n->GetType();
    if (type == DATA) {
      if (CheckInsertInputMemcpy(n, bypass_index)) {
        data_nodes.emplace_back(n);
      }
    } else if (type == NETOUTPUT) {
      if (output_node == nullptr) {
        output_node = n;
      } else {
        REPORT_INNER_ERROR("E19999", "While_body graph:%s exists multi NetOutput nodes, check invalid",
                           while_body->GetName().c_str());
        GELOGE(FAILED, "[Check][Param] while_body %s exists multi NetOutput nodes.", while_body->GetName().c_str());
        return FAILED;
      }
    }
  }
  if (output_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "While_body graph:%s has no output, check invalid",
                       while_body->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] while_body %s has no output.", while_body->GetName().c_str());
    return FAILED;
  }

  if ((InsertInputMemcpy(while_body, data_nodes) != SUCCESS) ||
      (InsertOutputMemcpy(while_body, output_node, bypass_index) != SUCCESS)) {
    GELOGE(FAILED, "[Insert][MemcpyNode] in while_body %s failed.", while_body->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Insert input memcpy node in while_body
 * @param [in] graph: while_body
 * @param [in] data_nodes: data_nodes
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::InsertInputMemcpy(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes) {
  if (data_nodes.empty()) {
    GELOGD("No need to insert input memcpy node in while_body %s.", graph->GetName().c_str());
    return SUCCESS;
  }

  std::string in_name = graph->GetName() + "_input_Memcpy";
  OpDescBuilder in_builder(in_name, IDENTITY);
  for (size_t i = 0; i < data_nodes.size(); i++) {
    // Data node has and only has one output
    in_builder.AddInput("x" + std::to_string(i), data_nodes[i]->GetOpDesc()->GetOutputDesc(0))
              .AddOutput("y"  + std::to_string(i), data_nodes[i]->GetOpDesc()->GetOutputDesc(0));
  }
  GELOGD("Insert memcpy after data_nodes of while_body %s.", graph->GetName().c_str());
  NodePtr in_memcpy = graph->AddNode(in_builder.Build());
  GE_CHECK_NOTNULL(in_memcpy);
  for (size_t i = 0; i < data_nodes.size(); i++) {
    // Data node has and only has one output
    OutDataAnchorPtr out_data_anchor = data_nodes[i]->GetOutDataAnchor(0);
    std::vector<InDataAnchorPtr> in_anchors;
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      in_anchors.emplace_back(peer_in_anchor);
    }
    if (InsertNodeBetween(out_data_anchor, in_anchors, in_memcpy, i, i) != SUCCESS) {
      GELOGE(FAILED, "[Insert][MemcpyAsync] %s in while_body %s failed.", in_name.c_str(), graph->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Insert output memcpy node in while_body
 * @param [in] graph: while_body
 * @param [in] output_node: NetOutput
 * @param [in] bypass_index
 * @return: 0 for SUCCESS / others for FAILED
 */
Status SubgraphPass::InsertOutputMemcpy(const ComputeGraphPtr &graph, const NodePtr &output_node,
                                        const std::set<uint32_t> &bypass_index) {
  if (output_node->GetAllInDataAnchorsSize() == bypass_index.size()) {
    GELOGD("No need to insert output memcpy node in while_body %s, output_size=%u, bypass_num=%zu.",
           graph->GetName().c_str(), output_node->GetAllInDataAnchorsSize(), bypass_index.size());
    return SUCCESS;
  }

  std::string out_name = graph->GetName() + "_output_Memcpy";
  OpDescBuilder out_builder(out_name, IDENTITY);
  for (size_t i = 0; i < output_node->GetAllInDataAnchorsSize(); i++) {
    if (bypass_index.count(i) == 0) {
      out_builder.AddInput("x" + std::to_string(i), output_node->GetOpDesc()->GetInputDesc(i))
                 .AddOutput("y" + std::to_string(i), output_node->GetOpDesc()->GetInputDesc(i));
    }
  }
  GELOGD("Insert memcpy before NetOutput of while_body %s.", graph->GetName().c_str());
  NodePtr out_memcpy = graph->AddNode(out_builder.Build());
  GE_CHECK_NOTNULL(out_memcpy);
  size_t cnt = 0;
  for (size_t i = 0; i < output_node->GetAllInDataAnchorsSize(); i++) {
    if (bypass_index.count(i) == 0) {
      InDataAnchorPtr in_data_anchor = output_node->GetInDataAnchor(i);
      OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      if (InsertNodeBetween(peer_out_anchor, {in_data_anchor}, out_memcpy, cnt, cnt) != SUCCESS) {
        GELOGE(FAILED, "[Insert][MemcpyAsync] %s in while_body %s failed.", out_name.c_str(), graph->GetName().c_str());
        return FAILED;
      }
      cnt++;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Check is data->netoutput without change in while body
 * @param [in] node: data node
 * @param [out] bypass_index
 * @return: false for data->netoutput without change in while body / for true for others
 */
bool SubgraphPass::CheckInsertInputMemcpy(const NodePtr &node, std::set<uint32_t> &bypass_index) {
  uint32_t input_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, input_index)) {
    return true;
  }

  // Data node has and only has one output
  OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(0);
  if ((out_data_anchor == nullptr) || (out_data_anchor->GetPeerInDataAnchors().size() != 1)) {
    return true;
  }
  InDataAnchorPtr peer_in_anchor = out_data_anchor->GetPeerInDataAnchors().at(0);
  if (peer_in_anchor->GetOwnerNode()->GetType() != NETOUTPUT) {
    return true;
  }

  OpDescPtr op_desc = peer_in_anchor->GetOwnerNode()->GetOpDesc();
  uint32_t output_index = 0;
  if ((op_desc == nullptr) ||
      !AttrUtils::GetInt(op_desc->GetInputDesc(peer_in_anchor->GetIdx()), ATTR_NAME_PARENT_NODE_INDEX, output_index)) {
    return true;
  }

  if (input_index != output_index) {
    return true;
  }
  bypass_index.insert(peer_in_anchor->GetIdx());
  return false;
}

/**
 * @ingroup ge
 * @brief Check is AtomicOp->NetOutput
 * @param [in] node
 * @param [in] out_index
 * @return: true for AtomicOp->NetOutput / false for others
 */
bool SubgraphPass::IsAtomicRequired(const NodePtr &node, int64_t out_index) {
  auto op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool is_atomic = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
    if (is_atomic) {
      std::vector<int64_t> atomic_output_index;
      // If GetListInt fail, atomic_output_index is empty.
      (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
      for (int64_t ind : atomic_output_index) {
        if (ind == out_index) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Check is OutputContinuesRequiredOp->NetOutput
 * @param [in] node
 * @return: true for OutputContinuesRequiredOp->NetOutput / false for others
 */
bool SubgraphPass::IsOutputContinuesRequired(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool continuous_output_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, continuous_output_flag);
    bool no_padding_continuous_output_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, no_padding_continuous_output_flag);
    return continuous_output_flag || no_padding_continuous_output_flag;
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Check is InputContinuesRequiredOp->NetOutput
 * @param [in] node
 * @return: true for InputContinuesRequiredOp->NetOutput / false for others
 */
bool SubgraphPass::IsInputContinuesRequired(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    bool continuous_input_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, continuous_input_flag);
    bool no_padding_continuous_input_flag = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, no_padding_continuous_input_flag);
    return continuous_input_flag || no_padding_continuous_input_flag;
  }
  return false;
}

/**
 * @ingroup ge
 * @brief Insert memcpy node
 * @param [in] graph
 * @param [in] out_anchor
 * @param [in] in_anchors
 * @param [in] name
 * @return: 0 for success / others for fail
 */
Status SubgraphPass::InsertMemcpyNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_anchor,
                                      const std::vector<InDataAnchorPtr> &in_anchors, const std::string &name) {
  GE_CHECK_NOTNULL(out_anchor);
  NodePtr in_node = out_anchor->GetOwnerNode();
  OpDescBuilder op_desc_builder(name, IDENTITY);
  OpDescPtr op_desc = op_desc_builder.AddInput("x", in_node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                     .AddOutput("y", in_node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()))
                                     .Build();
  (void)AttrUtils::SetBool(op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_CANNOT_BE_DELETED, true);
  if (GraphUtils::InsertNodeAfter(out_anchor, in_anchors, graph->AddNode(op_desc)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Insert Cast node %s(%s) after %s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                      out_anchor->GetOwnerNode()->GetName().c_str(),
                      out_anchor->GetOwnerNode()->GetType().c_str());
    GELOGE(FAILED, "[Insert][CastNode] %s(%s) after %s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           out_anchor->GetOwnerNode()->GetName().c_str(), out_anchor->GetOwnerNode()->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
/// @param [in] src
/// @param [in] dsts
/// @param [in] insert_node
/// @param [in] input_index
/// @param [in] output_index
/// @return Status
///
Status SubgraphPass::InsertNodeBetween(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                       const NodePtr &insert_node, uint32_t input_index, uint32_t output_index) {
  if (GraphUtils::AddEdge(src, insert_node->GetInDataAnchor(input_index)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
                      src->GetOwnerNode()->GetName().c_str(), src->GetOwnerNode()->GetType().c_str(), src->GetIdx(),
                      insert_node->GetName().c_str(), insert_node->GetType().c_str(), input_index);
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
           src->GetOwnerNode()->GetName().c_str(), src->GetOwnerNode()->GetType().c_str(), src->GetIdx(),
           insert_node->GetName().c_str(), insert_node->GetType().c_str(), input_index);
    return FAILED;
  }
  for (const auto &dst : dsts) {
    GELOGD("Insert node %s between %s->%s.", insert_node->GetName().c_str(), src->GetOwnerNode()->GetName().c_str(),
           dst->GetOwnerNode()->GetName().c_str());
    if ((GraphUtils::RemoveEdge(src, dst) != GRAPH_SUCCESS) ||
        (GraphUtils::AddEdge(insert_node->GetOutDataAnchor(output_index), dst) != GRAPH_SUCCESS)) {
      REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) or "
                        "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
                        src->GetOwnerNode()->GetName().c_str(), src->GetOwnerNode()->GetType().c_str(), src->GetIdx(),
                        dst->GetOwnerNode()->GetName().c_str(), dst->GetOwnerNode()->GetType().c_str(), dst->GetIdx(),
                        insert_node->GetName().c_str(), insert_node->GetType().c_str(), output_index,
                        dst->GetOwnerNode()->GetName().c_str(), dst->GetOwnerNode()->GetType().c_str(), dst->GetIdx());
      GELOGE(FAILED, "[Replace][DataEdge] %s:%d->%s:%d by %s:%u->%s:%d failed.",
             src->GetOwnerNode()->GetName().c_str(), src->GetIdx(),
             dst->GetOwnerNode()->GetName().c_str(), dst->GetIdx(),
             insert_node->GetName().c_str(), output_index,
             dst->GetOwnerNode()->GetName().c_str(), dst->GetIdx());
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace ge
