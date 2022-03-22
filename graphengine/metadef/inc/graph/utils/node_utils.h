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

#ifndef INC_GRAPH_UTILS_NODE_UTILS_H_
#define INC_GRAPH_UTILS_NODE_UTILS_H_

#include <set>
#include <map>
#include <vector>
#include "external/graph/operator.h"
#include "graph/anchor.h"
#include "graph/node.h"

namespace ge {
// Op types of Const like Opps.
extern const std::set<std::string> kConstOpTypes;

// Op types of Enter like Opps.
extern const std::set<std::string> kEnterOpTypes;
// Op types of Merge like Opps.
extern const std::set<std::string> kMergeOpTypes;
// Op types of Switch like Opps.
extern const std::set<std::string> kSwitchOpTypes;
// Op types of NextIteration like Opps.
extern const std::set<std::string> kNextIterationOpTypes;
// Op types of Exit like Opps.
extern const std::set<std::string> kExitOpTypes;

// Op types of If like Opps.
extern const std::set<std::string> kIfOpTypes;
// Op types of While like Opps.
extern const std::set<std::string> kWhileOpTypes;
// Op types of Case like Opps.
extern const std::set<std::string> kCaseOpTypes;
// Op types of For like Opps.
extern const std::set<std::string> kForOpTypes;

class NodeUtils {
 public:
  static graphStatus AddSendEventId(const NodePtr &node, const uint32_t &event_id);
  static graphStatus AddRecvEventId(const NodePtr &node, const uint32_t &event_id);
  static graphStatus GetSendEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_send);
  static graphStatus GetRecvEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_recv);

  static graphStatus ClearSendInfo();
  static graphStatus ClearRecvInfo();

  static graphStatus GetSingleOutputNodeOfNthLayer(const NodePtr &src, int depth, NodePtr &dst);

  static graphStatus GetDataOutAnchorAndControlInAnchor(const NodePtr &node_ptr, OutDataAnchorPtr &out_data,
                                                        InControlAnchorPtr &in_control);

  static graphStatus ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor);
  static graphStatus SetAllAnchorStatus(const NodePtr &node_ptr);
  static graphStatus SetAllAnchorStatus(Node &node);
  static bool IsAnchorStatusSet(const NodePtr &node_ptr);
  static bool IsAnchorStatusSet(const Node &node);

  static graphStatus MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node);

  static void UpdateIsInputConst(const NodePtr &node_ptr);
  static void UpdateIsInputConst(Node &node);
  static bool IsConst(const Node &node);
  static void UnlinkAll(const Node &node);
  static graphStatus UpdatePeerNodeInputDesc(const NodePtr &node_ptr);

  static graphStatus AppendInputAnchor(const NodePtr &node, uint32_t num);
  static graphStatus RemoveInputAnchor(const NodePtr &node, uint32_t num);

  static graphStatus AppendOutputAnchor(const NodePtr &node, uint32_t num);
  static graphStatus RemoveOutputAnchor(const NodePtr &node, uint32_t num);

  static bool IsInNodesEmpty(const Node &node);
  static GeTensorDesc GetOutputDesc(const Node &node, uint32_t index);
  static GeTensorDesc GetInputDesc(const Node &node, uint32_t index);
  static graphStatus UpdateOutputShape(const Node &node, uint32_t index, const GeShape &shape);
  static graphStatus UpdateInputShape(const Node &node, uint32_t index, const GeShape &shape);
  // check node whether unknown shape.If node shape contain -1 or -2,out param "is_unknow" will be true;
  // for func op, it will check subgraph yet, if some node shape of subgraph contain -1 or -2,
  // the out param "is_unknow" will be true too
  static graphStatus GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow);

  static std::string GetNodeType(const Node &node);
  static std::string GetNodeType(const NodePtr &node);

  static std::vector<ComputeGraphPtr> GetAllSubgraphs(const Node &node);
  static graphStatus GetDirectSubgraphs(const NodePtr &node, std::vector<ComputeGraphPtr> &subgraphs);
  static ComputeGraphPtr GetSubgraph(const Node &node, uint32_t index);
  static graphStatus SetSubgraph(Node &node, uint32_t index, const ComputeGraphPtr &subgraph);
  static NodePtr CreatNodeWithoutGraph(const OpDescPtr op_desc);
  ///
  /// Check if node is input of subgraph
  /// @param [in] node
  /// @return bool
  ///
  static bool IsSubgraphInput(const NodePtr &node);

  ///
  /// Check if node is output of subgraph
  /// @param [in] node
  /// @return bool
  ///
  static bool IsSubgraphOutput(const NodePtr &node);

  ///
  /// @brief Get subgraph original input node.
  /// @param [in] node
  /// @return Node
  ///
  static NodePtr GetParentInput(const Node &node);
  static NodePtr GetParentInput(const NodePtr &node);
  ///
  /// @brief Get subgraph original input node and corresponding out_anchor.
  /// @param [in] node
  /// @return NodeToOutAnchor  node and out_anchor which linked to in_param node
  ///
  static NodeToOutAnchor GetParentInputAndAnchor(const NodePtr &node);

  ///
  /// @brief Get is dynamic shape graph from node.
  /// @param [in] node
  /// @return bool
  ///
  static bool IsDynamicShape(const Node &node);
  static bool IsDynamicShape(const NodePtr &node);

  ///
  /// @brief Check is varying_input for while node
  /// @param [in] node: Data node for subgraph
  /// @return bool
  ///
  static bool IsWhileVaryingInput(const ge::NodePtr &node);

  ///
  /// @brief Get subgraph input is constant.
  /// @param [in] node
  /// @param [out] string
  /// @return bool
  ///
  static bool GetConstOpType(const NodePtr &node, std::string &type);

  ///
  /// @brief Remove node-related subgraphs, including subgraphs of nodes in the subgraph.
  /// @param [in] node
  /// @return return GRAPH_SUCCESS if remove successfully, other for failed.
  ///
  static graphStatus RemoveSubgraphsOnNode(const NodePtr &node);

  ///
  /// @brief Get subgraph input data node by index.
  /// @param [in] node
  /// @return Node
  ///
  static std::vector<NodePtr> GetSubgraphDataNodesByIndex(const Node &node, int index);

  ///
  /// @brief Get subgraph input data node by index.
  /// @param [in] node
  /// @return Node
  ///
  static std::vector<NodePtr> GetSubgraphOutputNodes(const Node &node);

  static NodePtr GetInDataNodeByIndex(const Node &node, const int index);

  static std::vector<std::pair<InDataAnchorPtr, NodePtr>> GetOutDataNodesWithAnchorByIndex(const Node &node, const int index);

  static ge::ConstNodePtr GetNodeFromOperator(const Operator &oprt);

  ///
  /// @brief Get node type in cross subgragh.
  /// @param [in] node
  /// @return type
  ///
  static std::string GetInConstNodeTypeCrossSubgraph(const ge::NodePtr &node);

  ///
  /// @brief Get node in cross subgragh.
  /// @param [in] node
  /// @return Node
  ///
  static NodePtr GetInNodeCrossSubgraph(const ge::NodePtr &node);

  ///
  /// @brief Get peer input node, supported get cross PartitionedCall .
  /// @param [in] node, current node
  /// @param [in] index, current node the index'th input, if it is PartionedCall's subgraph Data, please assign 0
  /// @param [out] peer_node,
  ///          A(PartionedCall_0)->B(PartionedCall_1)
  ///          PartionedCall_0's subgraph: Data->A->Netoutput
  ///          PartionedCall_1's subgraph: Data1->B->Netoutput
  ///          If it is called like GetInNodeCrossPartionCallNode(B,0,peer_node)or(Data1,0,peer_node), peer_node is A
  /// @return [graphStatus] running result of this function
  ///
  static graphStatus GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node);

  static graphStatus SetNodeParallelGroup(Node &node, const char *group_name);

  static graphStatus UpdateInputOriginalShapeAndShape(const Node &node, uint32_t index, const GeShape &shape);
  static graphStatus UpdateOutputOriginalShapeAndShape(const Node &node, uint32_t index, const GeShape &shape);

private:
  static std::map<NodePtr, std::vector<uint32_t>> map_send_info_;
  static std::map<NodePtr, std::vector<uint32_t>> map_recv_info_;
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_NODE_UTILS_H_
