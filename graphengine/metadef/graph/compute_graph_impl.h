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

#ifndef GRAPH_COMPUTE_GRAPH_IMPL_H_
#define GRAPH_COMPUTE_GRAPH_IMPL_H_

#include "graph/compute_graph.h"

namespace ge {
class ComputeGraphImpl {
 public:
  using ConstComputeGraphPtr  = std::shared_ptr<ConstComputeGraph>;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  explicit ComputeGraphImpl(const std::string &name);

  //ComputeGraphImpl(const ComputeGraphImpl& compute_graph) = default;
  ~ComputeGraphImpl() = default;

  std::string GetName() const;
  void SetName(const std::string &name);

  size_t GetAllNodesSize(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetAllNodes(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetAllNodes(const NodeFilter &node_filter,
                              const GraphFilter &graph_filter,
                              const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs,
                                const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetNodes(const bool is_unknown_shape,
                           const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetNodes(const bool is_unknown_shape,
                           const NodeFilter &node_filter,
                           const GraphFilter &graph_filter,
                           const ConstComputeGraphPtr &compute_graph) const;
  size_t GetDirectNodesSize() const;
  Vistor<NodePtr> GetDirectNode(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetInputNodes(const ConstComputeGraphPtr &compute_graph) const;
  Vistor<NodePtr> GetOutputNodes(const ConstComputeGraphPtr &compute_graph) const;
  NodePtr FindNode(const std::string &name) const;
  NodePtr FindFirstNodeMatchType(const std::string &name) const;

  bool GraphAttrsAreEqual(const ComputeGraphImpl &r_graph) const;
  bool VectorInputNodePtrIsEqual(const std::vector<NodePtr> &left_nodes, const std::vector<NodePtr> &right_nodes) const;
  bool GraphMembersAreEqual(const ComputeGraphImpl &r_graph) const;

  bool operator==(const ComputeGraphImpl &r_graph) const;

  NodePtr AddNodeFront(const NodePtr node);
  NodePtr AddNodeFront(const OpDescPtr &op, const ComputeGraphPtr &compute_graph);
  NodePtr AddNode(NodePtr node);
  NodePtr AddNode(OpDescPtr op, const ComputeGraphPtr &compute_graph);
  NodePtr AddNode(OpDescPtr op, const int64_t id, const ComputeGraphPtr &compute_graph);
  NodePtr AddInputNode(const NodePtr node);
  NodePtr AddOutputNode(const NodePtr node);
  NodePtr AddOutputNodeByIndex(const NodePtr node, const int32_t index);

  graphStatus RemoveConstInput(const NodePtr &node);
  graphStatus RemoveNode(const NodePtr &node);
  graphStatus RemoveInputNode(const NodePtr &node);
  graphStatus RemoveOutputNode(const NodePtr &node);

  std::shared_ptr<ComputeGraph> AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);
  graphStatus RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph);
  graphStatus AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph);
  void RemoveSubgraph(const std::string &name);

  std::shared_ptr<ComputeGraph> GetSubgraph(const std::string &name) const;
  std::vector<std::shared_ptr<ComputeGraph>> GetAllSubgraphs() const;
  void SetAllSubgraphs(const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs);

  shared_ptr<ComputeGraph> GetParentGraph();
  void SetParentGraph(const shared_ptr<ComputeGraph> &parent);
  shared_ptr<Node> GetParentNode();
  void SetParentNode(const shared_ptr<Node> &parent);

  const std::map<std::string, std::vector<int32_t>> &GetGraphOutNodes() const { return out_nodes_map_; }

  void SetOrigGraph(const ComputeGraphPtr &orig_graph) { origGraph_ = orig_graph; }
  ComputeGraphPtr GetOrigGraph(void) { return origGraph_; }
  void SetOutputSize(const uint32_t size) { output_size_ = size; }
  uint32_t GetOutputSize() const { return output_size_; }
  void SetInputSize(const uint32_t size) { input_size_ = size; }
  uint32_t GetInputSize() const { return input_size_; }

  // false: known shape  true: unknow shape
  bool GetGraphUnknownFlag() const { return is_unknown_shape_graph_; }
  void SetGraphUnknownFlag(const bool flag) { is_unknown_shape_graph_ = flag; }
  void SetNeedIteration(const bool need_iteration) { need_iteration_ = need_iteration; }
  bool GetNeedIteration() const { return need_iteration_; }

  const std::map<std::vector<std::string>, std::vector<std::string>> &GetShareParamLayer() const {
    return params_share_map_;
  }
  void SetShareParamLayer(const std::map<std::vector<std::string>, std::vector<std::string>> &params_share_map) {
    params_share_map_ = params_share_map;
  }

  void SetInputsOrder(const std::vector<std::string> &inputs_order) { inputs_order_ = inputs_order; }
  void SetGraphOutNodes(const std::map<std::string, std::vector<int32_t>> &out_nodes_map) { out_nodes_map_ = out_nodes_map; }
  void AppendGraphOutNodes(const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
    for (auto &item : out_nodes_map) {
      (void)out_nodes_map_.emplace(item.first, item.second);
    }
  }

  void SetGraphOpName(const std::map<uint32_t, std::string> &op_name_map) { op_name_map_ = op_name_map; }
  const std::map<uint32_t, std::string> &GetGraphOpName() const { return op_name_map_; }
  void SetAllNodesInfo(const std::map<OperatorImplPtr, NodePtr> &nodes) { all_nodes_infos_ = nodes; }

  void SetGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
    output_nodes_info_ = out_nodes_info;
  }

  void AppendGraphOutNodesInfo(std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
    (void)output_nodes_info_.insert(output_nodes_info_.end(), out_nodes_info.begin(), out_nodes_info.end());
  }

  const std::vector<std::pair<NodePtr, int32_t>> &GetGraphOutNodesInfo() const { return output_nodes_info_; }

  void SetGraphTargetNodesInfo(const std::vector<NodePtr> &target_nodes_info) {
    target_nodes_info_ = target_nodes_info;
  }
  const std::vector<NodePtr> &GetGraphTargetNodesInfo() const { return target_nodes_info_; }

  void SetSessionID(const uint64_t session_id) { session_id_ = session_id; }
  uint64_t GetSessionID() const { return session_id_; }

  void SetGraphID(const uint32_t graph_id) { graph_id_ = graph_id; }
  uint32_t GetGraphID() const { return graph_id_; }

  void SaveDataFormat(const ge::Format data_format) { data_format_ = data_format; }
  ge::Format GetDataFormat() const { return data_format_; }
  bool IsSummaryGraph() const { return is_summary_graph_; }
  void SetSummaryFlag(const bool is_summary_graph) { is_summary_graph_ = is_summary_graph; }

  graphStatus UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);
  graphStatus UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);
  graphStatus ReorderEventNodes(const ConstComputeGraphPtr &compute_graph);
  graphStatus InsertGraphEvents(const ConstComputeGraphPtr &compute_graph);

  graphStatus DFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                    std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::vector<NodePtr> &stack, const bool reverse,
                                    const ConstComputeGraphPtr &compute_graph);
  graphStatus BFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                    std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::deque<NodePtr> &stack,
                                    const ConstComputeGraphPtr &compute_graph);
  graphStatus CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                    std::map<std::string, NodePtr> &breadth_node_map);
  void TopologicalSorting(std::function<bool (const NodePtr &, const NodePtr &)> comp);
  graphStatus TopologicalSorting(const ComputeGraphPtr &const_graph_ptr,
                                 const ConstComputeGraphPtr &const_compute_graph);
  graphStatus TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph,
                                      const bool dfs_reverse = false);
  graphStatus SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num,
                        const ConstComputeGraphPtr &compute_graph);

  size_t GetInEdgeSize(const NodePtr &node);
  size_t GetOutEdgeSize(const NodePtr &node);

  bool IsValid() const;
  void InValid();
  void Dump(const ConstComputeGraphPtr &graph) const;
  void Swap(ComputeGraphImpl &graph);

  void SetNodesOwner(const ComputeGraphPtr &compute_graph);
  graphStatus IsolateNode(const NodePtr &node);
  graphStatus RemoveExtraOutEdge(const NodePtr &node);
  graphStatus Verify(const ConstComputeGraphPtr compute_graph);

  graphStatus InferShapeInNeed(const ComputeGraphPtr &const_graph_ptr,
                               const ConstComputeGraphPtr &const_compute_graph);

  ProtoAttrMap &MutableAttrMap();
  ConstProtoAttrMap &GetAttrMap() const;

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const;
  void SetUserDefOutput(const std::string &output_name);
  const std::string GetOutput();

  void EraseFromNodeList(const std::list<NodePtr>::iterator &position);
  void InsertToNodeList(const std::list<NodePtr>::iterator &position, const NodePtr &node);

  void PushBackToNodeList(const NodePtr &node);

  void EmplaceBackToNodeList(const NodePtr &node);
  void ClearNodeList();

 private:
  friend class ModelSerializeImp;
  friend class GraphUtils;
  std::string name_;
  std::list<NodePtr> nodes_;
  uint32_t graph_id_ = 0U;
  AttrStore attrs_;
  size_t direct_nodes_size_ = 0UL;
  std::map<OperatorImplPtr, NodePtr> all_nodes_infos_;
  std::vector<NodePtr> target_nodes_info_;

  std::vector<NodePtr> input_nodes_;
  std::vector<std::string> inputs_order_;
  uint32_t input_size_ = 1U;
  std::map<std::string, std::vector<int32_t>> out_nodes_map_;
  uint32_t output_size_ = 1U;
  std::vector<std::pair<NodePtr, int32_t>> output_nodes_info_;

  std::vector<std::shared_ptr<ComputeGraph>> sub_graph_;
  std::map<std::string, std::shared_ptr<ComputeGraph>> names_to_subgraph_;
  std::weak_ptr<ComputeGraph> parent_graph_;
  std::weak_ptr<Node> parent_node_;

  // the members followed should not in the ComputeGraph class
  bool is_valid_flag_;
  bool is_summary_graph_ = false;
  // Indicates whether it is need iteration
  bool need_iteration_ = false;
  std::map<std::vector<std::string>, std::vector<std::string>> params_share_map_;
  // TaskIdx -> op_name Map
  std::map<uint32_t, std::string> op_name_map_;
  uint64_t session_id_ = 0UL;
  ge::Format data_format_ = ge::FORMAT_ND;
  // unknown graph indicator, default is false, mean known shape
  bool is_unknown_shape_graph_ = false;
  // Graph Before BFE
  ComputeGraphPtr origGraph_;
};
}  // namespace ge
#endif  // GRAPH_COMPUTE_GRAPH_IMPL_H_
