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

#ifndef GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_
#define GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
class DynamicShapePartitioner {
 public:
  // An cluster means set of nodes that can be merged in same partition,
  // Corresponding relationship between cluster type and node:
  // DATA:DATA, UNKNOWN_SHAPE:unknowshape, KNOWN_SHAPE:knowshape, NETOUTPUT:NETOUTPUT.
  class Cluster : public std::enable_shared_from_this<Cluster> {
   public:
    enum Type { DATA, INPUT_NODE, NETOUTPUT, STAGE, KNOWN_SHAPE, UNKNOWN_SHAPE };
    Cluster(size_t rank, Type type, NodePtr node, DynamicShapePartitioner *partitioner)
        : id_(rank), min_(rank), max_(rank), type_(type), partitioner_(partitioner) {
      nodes_.push_back(node);
    }
    ~Cluster() = default;
    std::string DebugString() const;
    // Basic bean functions
    size_t Id() const;
    void UpdateRank(size_t rank);
    bool IsData() const;
    bool IsKnownShape() const;
    bool IsUnknownShape() const;
    bool IsIndependent() const;
    bool IsNetOutput() const;
    std::vector<std::shared_ptr<Cluster>> Inputs() const;
    std::vector<std::shared_ptr<Cluster>> Outputs() const;
    bool IsInputNode() const;
    std::vector<NodePtr> Nodes() const;
    bool IsRefVariable() const;
    // Cluster modify functions
    void AddInput(std::shared_ptr<Cluster> in);
    void RemoveInput(std::shared_ptr<Cluster> in);
    void AddOutput(std::shared_ptr<Cluster> out);
    void RemoveOutput(std::shared_ptr<Cluster> out);
    // Merge other cluster to this cluster, Whether it leads to a ring or not
    // Merge src to dst means:
    // All links to src will break and link to dst instead
    // All nodes of src will change its owner to dst
    // Update max and min rank of dst
    void Merge(std::shared_ptr<Cluster> other);
    // Try merge other cluster to this cluster, ONLY if will not leads to a ring
    bool TryMerge(std::shared_ptr<Cluster> other);
    // Merge all clusters on path(s) from other to this
    std::vector<std::shared_ptr<Cluster>> MergeAllPathFrom(std::shared_ptr<Cluster> other);
    // Convert cluster to functioned call functions
    void AddFrameInput(InDataAnchorPtr anchor);
    void AddFrameOutput(OutDataAnchorPtr anchor);
    InDataAnchorPtr GetFrameInDataAnchor(InDataAnchorPtr anchor);
    OutDataAnchorPtr GetFrameOutDataAnchor(OutDataAnchorPtr anchor);
    InControlAnchorPtr GetFrameInControlAnchor();
    OutControlAnchorPtr GetFrameOutControlAnchor();
    Status BuildFrame();
    Status BuildPartitionFrame();
    Status CombinePartitionFrame();
    Status BuildPartitionSubgraph();
    // Clear resource and break circular dependency
    void Clear();
    bool IsAdjoinNodes(const std::shared_ptr<Cluster> &other) const {
      const auto &out_clusters = other->out_clusters_;
      return std::find(out_clusters.begin(), out_clusters.end(), shared_from_this()) != out_clusters.end();
    }

   private:
    static thread_local size_t unique_id_;
    size_t id_;
    // Each Cluster records the maximum and minimum topological order of its node
    size_t min_;  // maximum topological order
    size_t max_;  // minimum topological order
    Type type_;
    std::vector<std::shared_ptr<Cluster>> in_clusters_;
    std::vector<std::shared_ptr<Cluster>> out_clusters_;
    std::vector<NodePtr> nodes_;
    // Fileds for build partitoned call and subgraph
    DynamicShapePartitioner *partitioner_;  // Not owned, the partitioner this cluster belongs to
    std::unordered_map<InDataAnchorPtr, size_t> inputs_index_;
    std::unordered_map<OutDataAnchorPtr, size_t> outputs_index_;
    std::vector<InDataAnchorPtr> inputs_;
    std::vector<OutDataAnchorPtr> outputs_;
    std::unordered_set<std::shared_ptr<Cluster>> control_inputs_;
    std::unordered_set<OutControlAnchorPtr> control_outputs_;
    NodePtr partition_node_;    // corresponding partitioned call node
    ComputeGraphPtr subgraph_;  // corresponding subgraph
  };
  explicit DynamicShapePartitioner(ge::ComputeGraphPtr graph) : root_graph_(graph) {}
  ~DynamicShapePartitioner() = default;

  Status Partition();

  using OrderedFilter = std::function<bool(const std::shared_ptr<Cluster> &cluster)>;

 private:
  Status PartitionImpl();
  // Collect nodes that satisfy the unknowshape rules:
  // 1) The Tensor shape of any input or output is unknow shape(dim_size = -1) or unknow rank(dim_size=-2)
  // 2) Subgraphs of the node has an operator that satisfies rule 1)
  Status MarkUnknownShapeNodes();
  // For each node a Cluster structure, and connected according to the connection relationship of the nodes
  // An cluster means set of nodes that can be merged in same partition,
  // Corresponding relationship between cluster type and node:
  // DATA:DATA, UNKNOWN_SHAPE:unknowshape, KNOWN_SHAPE:knowshape, NETOUTPUT:NETOUTPUT
  Status InitClusters();
  // Merge clusters according to the following rules:
  // 1) Iterate through the UNKNOWN_SHAPE clusters, if the input is UNKNOWN_SHAPE,
  //    merge all the clusters in the path(s) between the two clusters
  // 2) Iterate through the KNOWN_SHAPE clusters, if the input is KNOWN_SHAPE, and
  //    and there's only one path between the two clusters , merge the two clusters
  // 3) Iterate through the INPUT_DATA clusters, merge all INPUT_DATA
  Status MergeClusters();
  // Merge clusters step0
  void MergeClustersControlFlow();
  // Merge clusters step1
  void MergeClustersUnknownShape();
  // Merge clusters step2
  void MergeClustersKnownShape();
  // Merge clusters step3
  void MergeClustersInputData();
  // Topological sort clusters after merge unknown shape clusters.
  Status TopologicalSortClusters(const OrderedFilter &ordered_filter);
  // Deduplicate merged clusters
  void PruneUniqueClusters();
  // Establish the input-output anchors for each partition of the cluster and record links to other clusters
  Status BuildPartitionFrame();
  // Establish connection between corresponding partitioned of clusters
  Status CombinePartitionFrame();
  // Convert the nodes in cluster into a complete ComputeGraph
  Status BuildPartitionSubgraph();
  // Clear resource and break circular dependency
  void ClearResource();
  // Debug functions
  void DumpGraph(const std::string &suffix);
  std::string DebugString() const;
  bool JudgeUnknowShapeWithAttr(const OpDescPtr &opdesc);
  // Util functions
  Status CollectSpreadUnknownShapeNodes(NodePtr node);
  Status IsUnknownShapeGraph(ge::ComputeGraphPtr graph, bool &is_unknow);
  Status IsUnknownShapeNode(ge::NodePtr node, bool &is_unknow);
  Status CtrlEdgeTransfer();
  ge::ComputeGraphPtr root_graph_;                                        // The original graph to partition
  std::unordered_map<NodePtr, std::shared_ptr<Cluster>> node_2_cluster_;  // Record nodes and the cluster it belongs to
  // V1 control flow cluster, need merge to one Graph.
  std::map<int64_t, std::vector<std::shared_ptr<Cluster>>> control_clusters_;
  // topological sorted clusters, this field will change with the splitting.
  // When partitioning UNKNOWN_SHAPE cluster, it is a collection of all topological sorted UNKNOWN_SHAPE clusters
  // When partitioning KNOWN_SHAPE cluster, it is a collection of all topological sorted KNOWN_SHAPE clusters
  std::vector<std::shared_ptr<Cluster>> ordered_cluster_;
  // Unique clusters left after merged clusters
  std::unordered_set<std::shared_ptr<Cluster>> unique_clusters_;
  // Unique clusters left after merged clusters sorted by rank
  std::vector<std::shared_ptr<Cluster>> sorted_unique_clusters_;
  // Nodes of root_graph_ that satisfy the unknowshape rules
  std::unordered_set<NodePtr> unknown_shape_nodes_;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_DYNAMIC_SHAPE_PARTITION_H_
