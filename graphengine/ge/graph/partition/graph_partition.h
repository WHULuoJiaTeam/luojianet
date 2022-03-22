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

#ifndef GE_GRAPH_PARTITION_GRAPH_PARTITION_H_
#define GE_GRAPH_PARTITION_GRAPH_PARTITION_H_

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "external/graph/operator_reg.h"
#include "graph/partition/engine_place.h"

namespace ge {
using PartitionMap = std::unordered_map<ComputeGraphPtr, std::string>;
using NodetoNodeMap = std::unordered_map<NodePtr, NodePtr>;
using EnginetoGraphMap = std::unordered_map<std::string, ComputeGraphPtr>;
using EdgeMap = std::set<std::pair<AnchorPtr, AnchorPtr>>;
using ClusterSet = std::set<size_t>;
class Cluster {
 public:
  size_t index_;              // corresponding to rank of node
  ClusterSet in_clu_;         // inClusters index
  ClusterSet out_clu_;        // outClusters index
  std::list<NodePtr> nodes_;  // including node of this cluster
  std::string engine_name_;   // data like must be a specific engine
  std::string stream_label_;
  explicit Cluster(size_t index, std::string engine, std::string stream)
      : index_(index), engine_name_(std::move(engine)), stream_label_(std::move(stream)) {}
  ~Cluster() = default;
};
using ClusterPtr = std::shared_ptr<Cluster>;

class GraphPartitioner {
 public:
  /// Partition() can only be called in Partition mode.
  /// MergeAfterSubGraphOptimization() can only be called in Merge mode.
  /// After Partition(), change to Merge mode. After MergeAfterSubGraphOptimization(), change to Partition mode
  enum Mode { kPartitioning, kSecondPartitioning, kMerging };
  GraphPartitioner() : partition_times_(0){};
  ~GraphPartitioner() = default;

  // the main method that partitions the graph
  // input_size and output_size are the number of inputs and outputs in the original graph
  Status Partition(ComputeGraphPtr compute_graph, Mode mode);

  // after partition, all SubGraph will be merged back based on end<->pld.
  Status MergeAfterSubGraphOptimization(ComputeGraphPtr &output_merged_compute_graph,
                                        const ComputeGraphPtr &original_compute_graph);
  // Return all subgraphs
  const Graph2SubGraphInfoList &GetSubGraphMap();

  const Graph2InputNodesSubGraphInfo &GetSubGraphInfoMap() {return graph_2_input_subgraph_; }

 private:
  Status MergeSubGraph(ge::ComputeGraphPtr &output_merged_compute_graph,
                       const ge::ComputeGraphPtr &original_compute_graph);
  Status PartitionSubGraph(ge::ComputeGraphPtr compute_graph, Mode mode);
  Status MergeAllSubGraph(ComputeGraphPtr &output_merged_compute_graph,
                          const std::vector<SubGraphInfoPtr> &sub_graph_list);
  Status CheckIfEnd2PldEmpty(ComputeGraphPtr &output_merged_compute_graph);

  // Run engine placer, assign engine, check support amd init all clusters
  Status Initialize(ComputeGraphPtr compute_graph);

  /// add pld and end nodes between two sub-graphs for the specific anchors
  /// all anchors are in original graph
  Status AddPlaceHolderEnd(const AnchorPtr &out_anchor, const AnchorPtr &in_anchor);
  void AddNewGraphToPartition(ComputeGraphPtr &input_graph, const std::string &engine_name);
  Status AddPartitionsToGraphNode(vector<SubGraphInfoPtr> &output_subgraphs, ComputeGraphPtr compute_graph);

  // check if the node has no input
  bool HasNoInput(NodePtr node);

  // check if the node is data-like. Currently data-like means: data, variable, const
  bool IsDataLike(NodePtr node);

  // add place holder and end node in src and dst graph
  graphStatus AddPlaceHolderEndInSrcDstGraph(const AnchorPtr &out_data_anchor, const AnchorPtr &peer_in_anchor,
                                             const ComputeGraphPtr &pld_graph, const ComputeGraphPtr &end_graph);
  Status LinkInput2EndRemoveOrginalLink(NodePtr input_node, ComputeGraphPtr src_graph, ComputeGraphPtr dst_graph);

  /// After partition, put input nodes in srcGraph to dstGraph. Data will be linked to 'end';
  /// the other end will be linked to 'placeholder'
  Status PutInputNodesInSubGraph(const ComputeGraphPtr &src_graph, const ComputeGraphPtr &dst_graph);

  // Sort all subGraphs topologically, store the info in sorted_partitions_ <computeGraph, rank>
  Status SortSubGraphs(const ComputeGraphPtr &);
  AnchorPtr GetEndInAnchor(const AnchorPtr &src_anchor, const NodePtr &end_node);
  AnchorPtr GetPldOutAnchor(const NodePtr &pld_node, const AnchorPtr &dst_anchor);
  Status RemoveNodeAndEdgeBetweenEndPld(ComputeGraphPtr &output_merged_compute_graph,
                                        const std::vector<SubGraphInfoPtr> &sub_graph_list);
  void AddEndPldInformationToSubGraphInfo(SubGraphInfoPtr &sub_graph_info);
  bool IsMergeable(size_t parent_cluster, size_t child_cluster, size_t upper_bound);

  // Link from->to
  void InsertEdge(size_t from, size_t to);

  // Remove parent cluster's out and child cluster's in
  void RemoveEdge(size_t parent_cluster, size_t child_cluster);
  void MergeTwoClusters(size_t parent_cluster, size_t &child_cluster);

  // Check if there's a second path between two clusters. The max path length is upper_bound
  bool HasSecondPath(size_t src, size_t dst, size_t upper_bound);

  // Mark all clusters
  void MarkClusters();

  /// Split all sub graph and add placeholder, end according to marks
  /// traverse marked clusters and split them into sub-graphs
  Status SplitSubGraphs(ComputeGraphPtr compute_graph);
  Status UpdateEndOpDesc(const NodePtr &src_node, int output_index, OpDescPtr &end_op_desc);
  Status UpdatePldOpDesc(const NodePtr &dst_node, int input_index, OpDescPtr &end_op_desc);

  // Clear partition data
  void ClearAllPartitionData();
  void SetMergedGraphId(ComputeGraphPtr &output_merged_compute_graph);

  struct GraphPartitionInfo {
    EnginePlacer engine_placer_;
    PartitionMap partitions_;  // sub-graphs after partition <sub-graph-id, ComputeGraphPtr>
    std::unordered_map<ComputeGraphPtr, size_t> partitions_2_rank_;  // <subGraph, rank>
    std::vector<ComputeGraphPtr> rank_2_partitions_;                 // <rank, subGraph>
    NodetoNodeMap corresponding_node_in_partitions_;                 // mapping between a node in the original graph and
    uint32_t num_of_pld_end_;                                        // a counter to track 'place holder' and 'end'
    size_t input_size_;
    size_t output_size_;
    std::string output_name_;
    NodetoNodeMap end_2_pld_;                // mapping between each 'end; and 'placeHolder' node
    NodetoNodeMap pld_2_end_;                // mapping between each 'placeHolder' and 'end' node
    std::map<size_t, NodePtr> index_2_end_;  // order mapping between peerindex and 'end' node
    Mode mode_;
    std::unordered_map<size_t, ClusterPtr> clusters_;                       // index to cluster ptr, contains all nodes
    std::unordered_map<NodePtr, std::shared_ptr<Cluster>> node_2_cluster_;  // node map to cluster
    std::unordered_map<std::shared_ptr<Cluster>, ComputeGraphPtr> cluster_2_partition_;  // cluster map to subgraph
    void ClearAllData(Mode mode) {
      rank_2_partitions_.clear();
      partitions_2_rank_.clear();
      partitions_.clear();
      corresponding_node_in_partitions_.clear();
      index_2_end_.clear();
      cluster_2_partition_.clear();
      clusters_.clear();
      node_2_cluster_.clear();
      pld_2_end_.clear();
      end_2_pld_.clear();
      if (mode_ == kMerging) {
        mode_ = kPartitioning;
      } else {
        mode_ = mode;
      }
    }
    GraphPartitionInfo() : num_of_pld_end_(0), input_size_(0), output_size_(0), mode_(kPartitioning) {}
    ~GraphPartitionInfo() = default;
  };
  std::unordered_map<ComputeGraphPtr, GraphPartitionInfo> graph_2_graph_partition_info_;
  Graph2SubGraphInfoList graph_2_subgraph_list_;
  Graph2InputNodesSubGraphInfo graph_2_input_subgraph_;
  GraphPartitionInfo graph_info_;
  uint32_t partition_times_;  // times of call partition
  std::map<Mode, std::string> mode_2_str_ = {{kPartitioning, "Partitioning"},
    {kSecondPartitioning, "SecondPartitioning"}, {kMerging, "Merging"}};
  friend class GraphManager;
};
}  // namespace ge

#endif  // GE_GRAPH_PARTITION_GRAPH_PARTITION_H_
