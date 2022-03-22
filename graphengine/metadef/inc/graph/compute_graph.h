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

#ifndef INC_GRAPH_COMPUTE_GRAPH_H_
#define INC_GRAPH_COMPUTE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <list>
#include <deque>
#include "detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/range_vistor.h"

namespace ge {
using ConstComputeGraph = const ComputeGraph;

class OperatorImpl;
using OperatorImplPtr = std::shared_ptr<OperatorImpl>;

class ComputeGraphImpl;
using ComputeGraphImplPtr = std::shared_ptr<ComputeGraphImpl>;

using NodeFilter = std::function<bool(const Node &)>;
using GraphFilter = std::function<bool(const Node &, const char_t *, const ComputeGraphPtr &)>;

class ComputeGraph : public std::enable_shared_from_this<ComputeGraph>, public AttrHolder {
  friend class GraphUtils;

 public:
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  explicit ComputeGraph(const std::string &name);
  ~ComputeGraph() override;
  ComputeGraph(const ge::ComputeGraph& compute_graph);
  ComputeGraph(ge::ComputeGraph&& compute_graph);

  std::string GetName() const;
  void SetName(const std::string &name);

  using AttrHolder::DelAttr;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  size_t GetAllNodesSize() const;
  Vistor<NodePtr> GetAllNodes() const;
  // is_unknown_shape: false, same with GetAllNodes func
  // is_unknown_shape: true, same with GetDirectNodes func
  Vistor<NodePtr> GetNodes(bool is_unknown_shape) const;
  Vistor<NodePtr> GetNodes(bool is_unknown_shape, const NodeFilter &node_filter, const GraphFilter &graph_filter) const;
  size_t GetDirectNodesSize() const;
  Vistor<NodePtr> GetDirectNode() const;
  Vistor<NodePtr> GetInputNodes() const;
  Vistor<NodePtr> GetOutputNodes() const;

  NodePtr FindNode(const std::string &name) const;
  NodePtr FindFirstNodeMatchType(const std::string &name) const;
  // AddNode with NodePtr
  NodePtr AddNode(NodePtr node);
  NodePtr AddNode(OpDescPtr op);
  NodePtr AddNode(OpDescPtr op, const int64_t id);    // for unserialize
  NodePtr AddNodeFront(const NodePtr node);
  NodePtr AddNodeFront(const OpDescPtr &op);
  NodePtr AddInputNode(const NodePtr node);
  NodePtr AddOutputNode(const NodePtr node);
  NodePtr AddOutputNodeByIndex(const NodePtr node, const int32_t index);

  graphStatus RemoveNode(const NodePtr &node);
  graphStatus RemoveInputNode(const NodePtr &node);
  graphStatus RemoveOutputNode(const NodePtr &node);
  graphStatus RemoveConstInput(const NodePtr &node);

  /// Add a subgraph to this graph. The subgraph must has a parent graph and parent node,
  /// which means the member functions `SetParentGraph` and `SetParentNode` of the subgraph
  /// must be called before add it to the root graph. and subgraph->GetParentNode()->GetOwnerGraph()
  /// must equal to subgraph->GetOwnerGraph().
  /// The subgraphs can only be added to a *root graph*. A root graph is a graph without any parent graph.
  /// The subgraph's name SHOULD(not must) be the same as the parameter `name`
  graphStatus AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph);
  graphStatus AddSubgraph(const std::shared_ptr<ComputeGraph> &subgraph);

  void RemoveSubgraph(const std::string &name);
  void RemoveSubgraph(const std::shared_ptr<ComputeGraph> &subgraph);

  std::shared_ptr<ComputeGraph> GetSubgraph(const std::string &name) const;
  std::vector<std::shared_ptr<ComputeGraph>> GetAllSubgraphs() const;
  void SetAllSubgraphs(const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs);

  // obsolete
  std::shared_ptr<ComputeGraph> AddSubGraph(const std::shared_ptr<ComputeGraph> sub_graph);
  // obsolete
  graphStatus RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);

  ///
  /// @brief Update input-mapping
  /// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
  /// @return graphStatus
  ///
  graphStatus UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);

  ///
  /// @brief Update output-mapping
  /// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
  /// @return graphStatus
  ///
  graphStatus UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);

  void TopologicalSorting(std::function<bool (const NodePtr &, const NodePtr &)> comp);
  graphStatus TopologicalSorting();
  bool IsValid() const;
  void InValid();
  void Dump() const;

  void Swap(ComputeGraph &graph);

  graphStatus IsolateNode(const NodePtr &node);
  graphStatus Verify();
  graphStatus InferOriginFormat();
  graphStatus InferShapeInNeed();
  graphStatus InsertGraphEvents();
  bool operator==(const ComputeGraph &r_compute_graph) const;
  ComputeGraph& operator=(ge::ComputeGraph compute_graph);

  const std::map<std::vector<std::string>, std::vector<std::string>> &GetShareParamLayer() const;

  void SetShareParamLayer(const std::map<std::vector<std::string>, std::vector<std::string>> params_share_map);

  void SetInputsOrder(const std::vector<std::string> &inputs_order);

  void SetGraphOutNodes(const std::map<std::string, std::vector<int32_t>> out_nodes_map);

  void AppendGraphOutNodes(const std::map<std::string, std::vector<int32_t>> out_nodes_map);

  std::shared_ptr<ComputeGraph> GetParentGraph();
  void SetParentGraph(const shared_ptr<ComputeGraph> &parent);
  std::shared_ptr<Node> GetParentNode();
  void SetParentNode(const shared_ptr<Node> &parent);

  const std::map<std::string, std::vector<int32_t>> &GetGraphOutNodes() const;
  void SetOrigGraph(const ComputeGraphPtr orig_graph);

  ComputeGraphPtr GetOrigGraph(void);
  void SetOutputSize(const uint32_t size);
  uint32_t GetOutputSize() const;
  void SetInputSize(const uint32_t size);
  uint32_t GetInputSize() const;

  // false: known shape  true: unknow shape
  bool GetGraphUnknownFlag() const;
  void SetGraphUnknownFlag(const bool flag);

  ///
  /// Set is need train iteration.
  /// If set true, it means this graph need to be run iteration some
  /// times(according variant "npu_runconfig/iterations_per_loop").
  /// @param need_iteration is need iteration
  ///
  void SetNeedIteration(const bool need_iteration);

  void SetUserDefOutput(const std::string &output_name);

  const std::string GetOutput();

  ///
  /// Get is need train iteration.
  /// @return is need iteration
  ///
  bool GetNeedIteration() const;

  void SetGraphOpName(const std::map<uint32_t, std::string> &op_name_map);
  const std::map<uint32_t, std::string> &GetGraphOpName() const;

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const;

  void SetAllNodesInfo(const std::map<OperatorImplPtr, NodePtr> &nodes);

  void SetGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info);
  void AppendGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info);
  const std::vector<std::pair<NodePtr, int32_t>> &GetGraphOutNodesInfo() const;

  void SetGraphTargetNodesInfo(const std::vector<NodePtr> &target_nodes_info);
  const std::vector<NodePtr> &GetGraphTargetNodesInfo() const;

  void SetSessionID(const uint64_t session_id);
  uint64_t GetSessionID() const;

  void SetGraphID(const uint32_t graph_id);
  uint32_t GetGraphID() const;

  void SaveDataFormat(const ge::Format data_format);
  ge::Format GetDataFormat() const;
  bool IsSummaryGraph() const;
  void SetSummaryFlag(const bool is_summary_graph);

  /// nodes like : (a) <--- (c) ---> (b)
  /// node a and b have only one parent node c, and a is connected to c firstly
  /// topo order of DFS is `c, b, a` with `dfs_reverse=false` as default
  /// in same case, user could get `c, a, b` with `dfs_reverse=true`
  graphStatus TopologicalSortingGraph(const bool dfs_reverse = false);
  /**
   *  Move Send Event nodes after it`s control node
   *  Move Recv Event nodes before it`s control node
   */
  graphStatus ReorderEventNodes();

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  graphStatus DFSTopologicalSorting(std::vector<NodePtr> &node_vec, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::vector<NodePtr> &stack, const bool reverse);
  graphStatus BFSTopologicalSorting(std::vector<NodePtr> &node_vec, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::deque<NodePtr> &stack);
  graphStatus CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::map<string, NodePtr> &breadth_node_map);

  graphStatus SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &mapInEdgeNum);
  Vistor<NodePtr> AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs) const;
  Vistor<NodePtr> GetAllNodes(const NodeFilter &node_filter, const GraphFilter &graph_filter) const;
  size_t GetInEdgeSize(const NodePtr &node);
  size_t GetOutEdgeSize(const NodePtr &node);
  graphStatus RemoveExtraOutEdge(const NodePtr &node);
  bool GraphMembersAreEqual(const ComputeGraph &r_graph) const;
  bool GraphAttrsAreEqual(const ComputeGraph &r_graph) const;
  bool VectorInputNodePtrIsEqual(const std::vector<NodePtr> &r_node_ptr_vector,
                                 const std::vector<NodePtr> &l_node_ptr_vector) const;

  void SetNodesOwner();
  /**
   *  To improve preformace of list.size(), we should keep counter on nodes_.size()
   *  Use follow function to add/erase node from nodes_
   */
  void EraseFromNodeList(const std::list<NodePtr>::iterator position);

  void InsertToNodeList(const std::list<NodePtr>::iterator position, const NodePtr &node);

  void PushBackToNodeList(const NodePtr &node);

  void EmplaceBackToNodeList(const NodePtr &node);

  void ClearNodeList();

  friend class ModelSerializeImp;
  friend class GraphDebugImp;
  friend class OnnxUtils;
  friend class TuningUtils;

  ComputeGraphImplPtr impl_;
};
}  // namespace ge
#endif  // INC_GRAPH_COMPUTE_GRAPH_H_
