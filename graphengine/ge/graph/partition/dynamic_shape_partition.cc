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

#include "graph/partition/dynamic_shape_partition.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/omg_util.h"

#define REQUIRE(cond, ...)                                     \
  do {                                                         \
    if (!(cond)) {                                             \
      REPORT_INNER_ERROR("E19999", __VA_ARGS__);               \
      GELOGE(FAILED, "[Dynamic shape partition]" __VA_ARGS__); \
      return FAILED;                                           \
    }                                                          \
  } while (0)

#define REQUIRE_NOT_NULL(cond, ...) REQUIRE(((cond) != nullptr), __VA_ARGS__)
#define REQUIRE_SUCCESS(cond, ...) REQUIRE(((cond) == SUCCESS), __VA_ARGS__)
#define REQUIRE_GRAPH_SUCCESS(cond, ...) REQUIRE(((cond) == GRAPH_SUCCESS), __VA_ARGS__)

namespace ge {
using Cluster = DynamicShapePartitioner::Cluster;
using ClusterPtr = std::shared_ptr<Cluster>;

static bool IsSingleOpScene(const ComputeGraphPtr &root_graph) {
  for (const auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // not do partition in single op scene.
    bool is_singleop = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, is_singleop);
    if (is_singleop) {
      return true;
    }
  }
  return false;
}

Status DynamicShapePartitioner::Partition() {
  REQUIRE_NOT_NULL(root_graph_, "[Check][Param] Graph is nullptr.");
  if (IsSingleOpScene(root_graph_)) {
    GELOGD("Skip dynamic shape partition as in single op scene.");
    REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
            "[Set][Attr] dynamic shape partitioned flag on root graph:%s failed.", root_graph_->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start dynamic shape partition graph %s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(MarkUnknownShapeNodes(), "[Call][MarkUnknownShapeNodes] failed, root grah name:%s.",
                  root_graph_->GetName().c_str());
  if (unknown_shape_nodes_.empty()) {
    GELOGD("Skip dynamic shape partition of graph %s as all nodes are known shape.", root_graph_->GetName().c_str());
    REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
            "[Set][Attr] dynamic shape partitioned flag on root graph %s failed.", root_graph_->GetName().c_str());
    return SUCCESS;
  }
  REQUIRE(AttrUtils::SetBool(*root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true),
          "[Set][Attr] dynamic shape partitioned flag on root graph %s failed.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(CtrlEdgeTransfer(), "[Call][CtrlEdgeTransfer] failed, graph:%s.", root_graph_->GetName().c_str());
  DumpGraph("_Before_DSP");
  auto status = PartitionImpl();
  GELOGD("%s.", DebugString().c_str());
  if (status != SUCCESS) {
    GELOGE(status, "[Call][PartitionImpl] Failed dynamic shape partition graph:%s, ret:%s",
           root_graph_->GetName().c_str(), DebugString().c_str());
  }
  DumpGraph("_After_DSP");
  GELOGD("Finish dynamic shape partition graph %s.", root_graph_->GetName().c_str());
  ClearResource();
  return status;
}

Status DynamicShapePartitioner::CtrlEdgeTransfer() {
  GELOGD("Do ctrl edge transfer start!");
  GE_CHECK_NOTNULL(root_graph_);

  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (!is_dynamic_shape) {
    return SUCCESS;
  }
  for (auto &subgraph : root_graph_->GetAllSubgraphs()) {
    for (ge::NodePtr &n : subgraph->GetDirectNode()) {
      auto op_desc = n->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      auto op_type = op_desc->GetType();
      if (op_type == CONSTANT || op_type == CONSTANTOP) {
        if (n->GetInAllNodes().empty()) {
          GELOGD("[CtrlEdgeTransferPass] node [%s] in nodes is empty", n->GetName().c_str());
          continue;
        }

        GELOGD("start to tranfer ctrl edge for const node [%s]", n->GetName().c_str());

        for (auto &in_control_node : n->GetInControlNodes()) {
          GE_CHECK_NOTNULL(in_control_node);
          GE_CHK_STATUS_RET(ge::GraphUtils::RemoveEdge(in_control_node->GetOutControlAnchor(),
                                                       n->GetInControlAnchor()),
                            "[Remove][Edge] between %s and %s failed",
                            in_control_node->GetOutControlAnchor()->GetOwnerNode()->GetName().c_str(),
                            n->GetName().c_str());
          for (auto &out_node : n->GetOutNodes()) {
            if (out_node == nullptr) {
              continue;
            }
            GE_CHK_STATUS_RET(ge::GraphUtils::AddEdge(in_control_node->GetOutControlAnchor(),
                                                      out_node->GetInControlAnchor()),
                              "[Add][Edge] between %s and %s failed.",
                              in_control_node->GetOutControlAnchor()->GetOwnerNode()->GetName().c_str(),
                              out_node->GetName().c_str());
          }
        }
      }
    }
  }

  GELOGD("Do ctrl edge transfer end!");
  return SUCCESS;
}

Status DynamicShapePartitioner::PartitionImpl() {
  REQUIRE_SUCCESS(root_graph_->TopologicalSorting(),
                  "[Call][TopologicalSorting] failed, graph:%s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(InitClusters(), "[Init][Clusters] failed, graph:%s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(MergeClusters(), "[Merge][Clusters] failed, graph:%s.", root_graph_->GetName().c_str());
  PruneUniqueClusters();
  REQUIRE_SUCCESS(BuildPartitionFrame(), "[Build][PartitionFrame] failed, graph:%s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(CombinePartitionFrame(),
                  "[Combine][PartitionFrame] failed, graph:%s.", root_graph_->GetName().c_str());
  REQUIRE_SUCCESS(BuildPartitionSubgraph(),
                  "[Build][PartitionSubgraph] failed, graph:%s.", root_graph_->GetName().c_str());
  return SUCCESS;
}

void DynamicShapePartitioner::PruneUniqueClusters() {
  for (auto &node : root_graph_->GetDirectNode()) {
    auto cluster = node_2_cluster_[node];
    if (unique_clusters_.count(cluster) != 0) {
      continue;
    }
    if (unique_clusters_.insert(cluster).second) {
      sorted_unique_clusters_.emplace_back(cluster);
    }
  }
  auto comp_func = [](std::shared_ptr<Cluster> clu_a, std::shared_ptr<Cluster> clu_b) -> bool {
    return clu_a->Id() < clu_b->Id();
  };
  std::sort(sorted_unique_clusters_.begin(), sorted_unique_clusters_.end(), comp_func);
}

Status DynamicShapePartitioner::BuildPartitionFrame() {
  for (const auto &cluster : sorted_unique_clusters_) {
    REQUIRE_SUCCESS(cluster->BuildFrame(), "[Build][Frame] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::CombinePartitionFrame() {
  for (const auto &cluster : sorted_unique_clusters_) {
    REQUIRE_SUCCESS(cluster->CombinePartitionFrame(), "[Combine][Frame] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::BuildPartitionSubgraph() {
  for (const auto &cluster : sorted_unique_clusters_) {
    REQUIRE_SUCCESS(cluster->BuildPartitionSubgraph(), "[Build][SubGraph] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

std::string DynamicShapePartitioner::DebugString() const {
  size_t unknown = 0;
  size_t known = 0;
  size_t data = 0;
  size_t netoutput = 0;
  size_t is_inputnode = 0;
  size_t stage = 0;
  std::stringstream ss;
  ss << "All unknown shape nodes:" << std::endl;
  for (const auto &node : unknown_shape_nodes_) {
    ss << "  [" << node->GetName() << "](" << node->GetType() << ")" << std::endl;
  }
  for (const auto &cluster : unique_clusters_) {
    if (cluster->IsUnknownShape()) {
      unknown++;
    } else if (cluster->IsKnownShape()) {
      known++;
    } else if (cluster->IsData()) {
      data++;
    } else if (cluster->IsNetOutput()) {
      netoutput++;
    } else if (cluster->IsInputNode()) {
      is_inputnode++;
    } else if (cluster->IsIndependent()) {
      stage++;
    }
  }
  ss << "All clusters:" << unique_clusters_.size() << ", data:" << data << ", known:" << known
     << ", unknown:" << unknown << ", netoutput:" << netoutput << ", is_inputnode:" << is_inputnode
     << ", stage:" << stage << std::endl;
  for (const auto &cluster : unique_clusters_) {
    ss << "  " << cluster->DebugString() << std::endl;
  }
  return ss.str();
}

void DynamicShapePartitioner::DumpGraph(const std::string &suffix) {
  GraphUtils::DumpGEGraphToOnnx(*root_graph_, root_graph_->GetName() + suffix);
  for (const auto &sub_graph : root_graph_->GetAllSubgraphs()) {
    GraphUtils::DumpGEGraphToOnnx(*sub_graph, sub_graph->GetName() + suffix);
  }
}

void DynamicShapePartitioner::ClearResource() {
  for (const auto &cluster : unique_clusters_) {
    cluster->Clear();
  }
  node_2_cluster_.clear();
  ordered_cluster_.clear();
  unique_clusters_.clear();
  sorted_unique_clusters_.clear();
  unknown_shape_nodes_.clear();
  root_graph_.reset();
}

Status DynamicShapePartitioner::MarkUnknownShapeNodes() {
  for (auto &node : root_graph_->GetDirectNode()) {
    REQUIRE_SUCCESS(CollectSpreadUnknownShapeNodes(node),
                    "[Call][CollectSpreadUnknownShapeNodes] for node:%s failed.", node->GetName().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::InitClusters() {
  auto graph = root_graph_;
  size_t rank = 0;
  for (const auto &node : graph->GetDirectNode()) {
    Cluster::Type type = Cluster::DATA;
    bool is_input = ((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP)) && node->GetInNodes().empty();
    REQUIRE_NOT_NULL(node->GetOpDesc(), "[Get][OpDesc] op_desc is null, graph:%s", graph->GetName().c_str());
    if (node->GetType() == DATA) {
      type = Cluster::DATA;
    } else if (is_input) {
      type = Cluster::INPUT_NODE;
    } else if (node->GetType() == NETOUTPUT) {
      type = Cluster::NETOUTPUT;
    } else if ((node->GetType() == PARTITIONEDCALL) && (node->GetOpDesc()->HasAttr(ATTR_STAGE_LEVEL))) {
      type = Cluster::STAGE;
    } else if (unknown_shape_nodes_.count(node) > 0) {
      type = Cluster::UNKNOWN_SHAPE;
    } else {
      type = Cluster::KNOWN_SHAPE;
    }
    auto cluster = MakeShared<Cluster>(rank++, type, node, this);
    REQUIRE_NOT_NULL(cluster, "[New][Memory] for cluster failed.");
    node_2_cluster_[node] = cluster;

    int64_t group_index = -1;
    if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index)) {
      GELOGD("[%s] is rts control flow Op, group index: %ld", node->GetName().c_str(), group_index);
      auto &control_cluster = control_clusters_[group_index];
      control_cluster.emplace_back(cluster);
    }

    // Already sorted topologically, so access to the parent cluster is safe
    for (const auto &parent : node->GetInAllNodes()) {
      cluster->AddInput(node_2_cluster_[parent]);
    }
  }
  for (const auto &node : graph->GetDirectNode()) {
    GELOGD("Make cluster for node %s : %s.", node->GetName().c_str(), node_2_cluster_[node]->DebugString().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::TopologicalSortClusters(const OrderedFilter &ordered_filter) {
  ordered_cluster_.clear();
  // BFS topological sort clusters for known shape cluster
  std::queue<ClusterPtr> ready_clusters;
  std::unordered_map<ClusterPtr, size_t> cluster_pending_count;
  std::unordered_set<ClusterPtr> seen_clusters;
  for (auto &node : root_graph_->GetDirectNode()) {
    auto &cluster = node_2_cluster_[node];
    if (seen_clusters.count(cluster) != 0) {
      continue;
    }
    seen_clusters.insert(cluster);
    auto pending_count = cluster->Inputs().size();
    if (pending_count == 0) {
      ready_clusters.push(cluster);
    } else {
      cluster_pending_count[cluster] = pending_count;
    }
  }

  size_t rank = 0;
  while (!ready_clusters.empty()) {
    auto cluster = ready_clusters.front();
    ready_clusters.pop();
    cluster->UpdateRank(rank++);
    if (ordered_filter == nullptr || ordered_filter(cluster)) {
      ordered_cluster_.push_back(cluster);
    }
    for (const auto &out_cluster : cluster->Outputs()) {
      if (cluster_pending_count[out_cluster] > 0 && --cluster_pending_count[out_cluster] == 0) {
        ready_clusters.push(out_cluster);
      }
    }
  }
  if (rank != seen_clusters.size()) {
    return FAILED;
  }
  return SUCCESS;
}

namespace {
static std::string ToString(const std::vector<ClusterPtr> &clusters) {
  if (clusters.empty()) {
    return "()";
  }
  std::stringstream ss;
  ss << "(";
  auto iter = clusters.begin();
  for (size_t i = 0; i < clusters.size() - 1; i++) {
    ss << (*iter)->Id() << ",";
    iter++;
  }
  ss << (*iter)->Id() << ").";
  return ss.str();
}
}

void DynamicShapePartitioner::MergeClustersControlFlow() {
  std::unordered_set<ClusterPtr> all_merged_clusters;
  for (const auto &item : control_clusters_) {
    const auto &control_cluster = item.second;
    auto rit = control_cluster.rbegin();
    if (rit == control_cluster.rend()) {
      GELOGW("Invalid empty control flow cluster.");
      continue;
    }

    const auto &cluster = *rit;
    if (all_merged_clusters.count(cluster) > 0) {
      continue;
    }

    for (++rit; rit != control_cluster.rend(); ++rit) {
      const auto &cluster_from = *rit;
      if (all_merged_clusters.count(cluster_from) > 0) {
        continue;
      }

      auto merged_clusters = cluster->MergeAllPathFrom(cluster_from);
      GELOGD("Merge all path cluster from %lu to %lu %s.", cluster_from->Id(), cluster->Id(),
             ToString(merged_clusters).c_str());
      for (const auto &merged_cluster : merged_clusters) {
        all_merged_clusters.emplace(merged_cluster);
        for (const auto &node : merged_cluster->Nodes()) {
          node_2_cluster_[node] = cluster;
        }
      }
    }
  }
}

void DynamicShapePartitioner::MergeClustersUnknownShape() {
  // Merge unknown shape clusters
  for (const auto &cluster : ordered_cluster_) {
    if (cluster->IsIndependent()) {
      continue;
    }
    for (const auto &in_cluster : cluster->Inputs()) {
      if (!in_cluster->IsUnknownShape()) {
        continue;
      }
      if (!cluster->IsAdjoinNodes(in_cluster)) {
        continue;
      }
      auto merged_clusters = cluster->MergeAllPathFrom(in_cluster);
      GELOGD("Merge all path cluster from %lu to %lu %s.", in_cluster->Id(), cluster->Id(),
             ToString(merged_clusters).c_str());
      for (const auto &merged_cluster : merged_clusters) {
        for (const auto &node : merged_cluster->Nodes()) {
          node_2_cluster_[node] = cluster;
        }
      }
    }
  }
}

void DynamicShapePartitioner::MergeClustersKnownShape() {
  // Merge known shape clusters
  for (const auto &cluster : ordered_cluster_) {
    if (cluster->IsIndependent()) {
      continue;
    }
    if (cluster->IsRefVariable() && cluster->Inputs().size() == 1) {
      auto in_cluster = *(cluster->Inputs().begin());
      in_cluster->Merge(cluster);
      node_2_cluster_[*(cluster->Nodes().begin())] = in_cluster;
      continue;
    }

    for (const auto &in_cluster : cluster->Inputs()) {
      if (!in_cluster->IsKnownShape()) {
        continue;
      }
      if (cluster->TryMerge(in_cluster)) {
        GELOGD("Success merge known shape cluster from %lu to %lu.", in_cluster->Id(), cluster->Id());
        for (const auto &node : in_cluster->Nodes()) {
          node_2_cluster_[node] = cluster;
        }
      }
    }
  }
}

void DynamicShapePartitioner::MergeClustersInputData() {
  // Merge input clusters
  std::shared_ptr<Cluster> cluster_pre = nullptr;
  for (const auto &cluster : ordered_cluster_) {
    if (!cluster->IsInputNode()) {
      continue;
    }
    if (cluster_pre != nullptr) {
      cluster_pre->Merge(cluster);
    } else {
      cluster_pre = cluster;
    }
    GELOGD("Success merge input node cluster from %lu to %lu.", cluster->Id(), cluster->Id());
    for (const auto &node : cluster->Nodes()) {
      node_2_cluster_[node] = cluster_pre;
    }
  }
}

Status DynamicShapePartitioner::MergeClusters() {
  const auto filter_known = [](const ClusterPtr &cluster) {
    return cluster->IsKnownShape() || cluster->IsInputNode();
  };
  const auto filter_unknown = [](const ClusterPtr &cluster) {
    return cluster->IsUnknownShape();
  };

  MergeClustersControlFlow();
  REQUIRE_SUCCESS(TopologicalSortClusters(filter_unknown),
                  "[TopologicalSort][Clusters] after merge control flow clusters failed.");
  MergeClustersUnknownShape();
  REQUIRE_SUCCESS(TopologicalSortClusters(filter_known),
                  "[TopologicalSort][Clusters] after merge unknown shape clusters failed.");
  MergeClustersKnownShape();
  MergeClustersInputData();
  return SUCCESS;
}

bool DynamicShapePartitioner::JudgeUnknowShapeWithAttr(const OpDescPtr &opdesc) {
  bool is_forced_unknown = false;
  if (AttrUtils::GetBool(opdesc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_forced_unknown) && is_forced_unknown) {
    GELOGD("Collect node %s as unknown as it was marked unknown forcibly.", opdesc->GetName().c_str());
    return true;
  }

  bool forced_unknown = false;
  if (AttrUtils::GetBool(opdesc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown) && forced_unknown) {
    GELOGD("Collect node %s as unknown as it was marked force unknown node forcibly.", opdesc->GetName().c_str());
    return true;
  }
  return false;
}

Status DynamicShapePartitioner::CollectSpreadUnknownShapeNodes(NodePtr node) {
  if (unknown_shape_nodes_.count(node) > 0) {
    return SUCCESS;
  }
  auto opdesc = node->GetOpDesc();
  REQUIRE_NOT_NULL(opdesc, "[Get][OpDesc] Opdesc is nullptr.");
  // One can set 'ATTR_NAME_IS_UNKNOWN_SHAPE=true' on node so as to forcing the node flow into the unknown subgraph,
  // ignore the actual shape.
  if (JudgeUnknowShapeWithAttr(opdesc)) {
    unknown_shape_nodes_.insert(node);
    return SUCCESS;
  }

  size_t anchor_index = 0;
  bool is_unknown = false;
  for (auto &out_tensor : opdesc->GetAllOutputsDesc()) {
    if (IsUnknownShapeTensor(out_tensor)) {
      GELOGD("Collect node %s as unknown as output %lu is unknown.", node->GetName().c_str(), anchor_index);
      is_unknown = true;
      auto anchor = node->GetOutDataAnchor(static_cast<int>(anchor_index));
      for (const auto peer_anchor : anchor->GetPeerInDataAnchors()) {
        if (peer_anchor != nullptr) {
          GELOGD("Collect node %s as has unknown input from %s:%lu.", peer_anchor->GetOwnerNode()->GetName().c_str(),
                 node->GetName().c_str(), anchor_index);
          unknown_shape_nodes_.insert(peer_anchor->GetOwnerNode());
        }
      }
    }
    anchor_index++;
  }
  anchor_index = 0;
  for (auto &in_tensor : opdesc->GetAllInputsDesc()) {
    if (IsUnknownShapeTensor(in_tensor)) {
      GELOGD("Collect node %s as unknown as input %lu is unknown.", node->GetName().c_str(), anchor_index);
      is_unknown = true;
      auto anchor = node->GetInDataAnchor(static_cast<int>(anchor_index));
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor != nullptr) {
        GELOGD("Collect node %s as has unknown output to %s:%lu.", peer_anchor->GetOwnerNode()->GetName().c_str(),
               node->GetName().c_str(), anchor_index);
        unknown_shape_nodes_.insert(peer_anchor->GetOwnerNode());
      }
    }
    anchor_index++;
  }
  if (is_unknown) {
    unknown_shape_nodes_.insert(node);
  } else {
    auto graph = root_graph_;
    for (const auto &subgraph_name : opdesc->GetSubgraphInstanceNames()) {
      auto subgraph = graph->GetSubgraph(subgraph_name);
      REQUIRE_NOT_NULL(subgraph, "[Get][Subgraph] %s of node %s on root graph failed.", subgraph_name.c_str(),
                       node->GetName().c_str());
      bool is_graph_unknow = false;
      REQUIRE_SUCCESS(IsUnknownShapeGraph(subgraph, is_graph_unknow),
                      "[Call][IsUnknownShapeGraph] Failed check subgraph %s shape of node %s.",
                      subgraph_name.c_str(), node->GetName().c_str());
      if (is_graph_unknow) {
        GELOGD("Collect node %s as its subgraph %s is unknown.", node->GetName().c_str(), subgraph->GetName().c_str());
        unknown_shape_nodes_.insert(node);
        break;
      }
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeNode(NodePtr node, bool &is_unknown) {
  auto opdesc = node->GetOpDesc();
  auto graph = root_graph_;
  for (auto &out_tensor : opdesc->GetAllOutputsDesc()) {
    if (IsUnknownShapeTensor(out_tensor)) {
      GELOGD("Mark node %s unknown as unknown output.", node->GetName().c_str());
      is_unknown = true;
      return SUCCESS;
    }
  }
  for (auto &in_tensor : opdesc->GetAllInputsDesc()) {
    if (IsUnknownShapeTensor(in_tensor)) {
      GELOGD("Mark node %s unknown as unknown intput.", node->GetName().c_str());
      is_unknown = true;
      return SUCCESS;
    }
  }
  for (auto &subgraph_name : opdesc->GetSubgraphInstanceNames()) {
    auto subgraph = graph->GetSubgraph(subgraph_name);
    REQUIRE_NOT_NULL(subgraph, "[Get][Subgraph] %s of node %s on root graph failed.", subgraph_name.c_str(),
                     node->GetName().c_str());
    REQUIRE_SUCCESS(IsUnknownShapeGraph(subgraph, is_unknown),
                    "[Call][IsUnknownShapeGraph] Failed check subgraph %s shape of node %s.",
                    subgraph_name.c_str(), node->GetName().c_str());
    if (is_unknown) {
      GELOGD("Mark node %s unknown as unknown subgraph.", node->GetName().c_str());
      return SUCCESS;
    }
  }
  is_unknown = false;
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeGraph(ComputeGraphPtr graph, bool &is_unknown) {
  for (auto &node : graph->GetDirectNode()) {
    REQUIRE_SUCCESS(IsUnknownShapeNode(node, is_unknown),
                    "[Call][IsUnknownShapeNode]Failed check node %s shape on graph %s.",
                    node->GetName().c_str(), graph->GetName().c_str());
    if (is_unknown) {
      GELOGD("Mark graph %s unknown as contains unknown node %s.", graph->GetName().c_str(), node->GetName().c_str());
      return SUCCESS;
    }
  }
  return SUCCESS;
}

std::string Cluster::DebugString() const {
  std::stringstream ss;
  switch (type_) {
    case DATA:
      ss << "DATA";
      break;
    case INPUT_NODE:
      ss << "INPUT_NODE";
      break;
    case NETOUTPUT:
      ss << "NETOUTPUT";
      break;
    case UNKNOWN_SHAPE:
      ss << "UNKNOW";
      break;
    case KNOWN_SHAPE:
      ss << "KNOW";
      break;
    default:
      break;
  }
  ss << "[" << id_ << "](size:" << nodes_.size() << ")";
  ss << "(" << min_ << "," << max_ << ")(";
  for (const auto &cluster : in_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")->(";
  for (const auto &cluster : out_clusters_) {
    ss << cluster->id_ << ",";
  }
  ss << ")|";
  for (const auto &node : nodes_) {
    ss << (node->GetName() + "|");
  }
  return ss.str();
}

size_t Cluster::Id() const { return id_; }
void Cluster::UpdateRank(size_t rank) {
  max_ = rank;
  min_ = rank;
};
bool Cluster::IsData() const { return type_ == DATA; };
bool Cluster::IsKnownShape() const { return type_ == KNOWN_SHAPE; };
bool Cluster::IsUnknownShape() const { return type_ == UNKNOWN_SHAPE; };
bool Cluster::IsIndependent() const { return type_ == STAGE; };
bool Cluster::IsNetOutput() const { return type_ == NETOUTPUT; };
bool Cluster::IsInputNode() const { return type_ == INPUT_NODE; };
bool Cluster::IsRefVariable() const {
  if ((nodes_.size() == 1) && ((nodes_[0]->GetType() == VARIABLE) || (nodes_[0]->GetType() == VARIABLEV2))) {
    std::string ref_variable_name;
    return (AttrUtils::GetStr(nodes_[0]->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_variable_name) &&
            !ref_variable_name.empty());
  }
  return false;
}

void Cluster::AddInput(ClusterPtr in) {
  if (std::find(in_clusters_.begin(), in_clusters_.end(), in) != in_clusters_.end()) return;
  in_clusters_.insert(in_clusters_.end(), in);
  if (std::find(in->out_clusters_.begin(), in->out_clusters_.end(), shared_from_this()) != in->out_clusters_.end())
    return;
  in->out_clusters_.insert(in->out_clusters_.end(), shared_from_this());
};
void Cluster::RemoveInput(ClusterPtr in) {
  in_clusters_.erase(std::remove(in_clusters_.begin(), in_clusters_.end(), in), in_clusters_.end());
  in->out_clusters_.erase(std::remove(in->out_clusters_.begin(), in->out_clusters_.end(), shared_from_this()),
                          in->out_clusters_.end());
};
void Cluster::AddOutput(ClusterPtr out) {
  if (std::find(out_clusters_.begin(), out_clusters_.end(), out) != out_clusters_.end()) return;
  out_clusters_.insert(out_clusters_.end(), out);
  if (std::find(out->in_clusters_.begin(), out->in_clusters_.end(), shared_from_this()) != out->in_clusters_.end())
    return;
  out->in_clusters_.insert(out->in_clusters_.end(), shared_from_this());
};
void Cluster::RemoveOutput(ClusterPtr out) {
  out_clusters_.erase(std::remove(out_clusters_.begin(), out_clusters_.end(), out), out_clusters_.end());
  out->in_clusters_.erase(std::remove(out->in_clusters_.begin(), out->in_clusters_.end(), shared_from_this()),
                          out->in_clusters_.end());
};
void Cluster::Merge(ClusterPtr other) {
  if (other->IsIndependent()) {
    return;
  }
  nodes_.insert(nodes_.end(), other->nodes_.begin(), other->nodes_.end());
  other->in_clusters_.erase(std::remove(other->in_clusters_.begin(), other->in_clusters_.end(), shared_from_this()),
                            other->in_clusters_.end());
  other->out_clusters_.erase(std::remove(other->out_clusters_.begin(), other->out_clusters_.end(), shared_from_this()),
                             other->out_clusters_.end());
  in_clusters_.erase(std::remove(in_clusters_.begin(), in_clusters_.end(), other), in_clusters_.end());
  out_clusters_.erase(std::remove(out_clusters_.begin(), out_clusters_.end(), other), out_clusters_.end());
  auto in_clusters = other->in_clusters_;
  for (const auto &cluster : in_clusters) {
    cluster->RemoveOutput(other);
    cluster->AddOutput(shared_from_this());
  }
  auto out_clusters = other->out_clusters_;
  for (const auto &cluster : out_clusters) {
    cluster->RemoveInput(other);
    cluster->AddInput(shared_from_this());
  }
  if (other->max_ > max_) {
    max_ = other->max_;
  }
  if (other->min_ < min_) {
    min_ = other->min_;
  }

  if (!IsUnknownShape() && other->IsUnknownShape()) {
    type_ = UNKNOWN_SHAPE;
  }
}

bool Cluster::TryMerge(ClusterPtr other) {
  std::queue<ClusterPtr> forward_reached;
  forward_reached.push(other);
  while (!forward_reached.empty()) {
    auto current_cluster = forward_reached.front();
    forward_reached.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if (cluster->max_ == max_ && current_cluster != other) {
        return false;
      } else if (cluster->min_ < max_) {
        forward_reached.push(cluster);
      }
    }
  }
  Merge(other);
  return true;
};
std::vector<ClusterPtr> Cluster::MergeAllPathFrom(ClusterPtr other) {
  std::queue<ClusterPtr> forward_reached_queue;
  std::queue<ClusterPtr> backward_reached_queue;

  std::unordered_set<ClusterPtr> forward_reached_clusters;
  std::unordered_set<ClusterPtr> backward_reached_clusters;
  std::vector<ClusterPtr> path_clusters;
  if (other->IsIndependent()) {
    return path_clusters;
  }

  path_clusters.push_back(other);
  forward_reached_queue.push(other);
  backward_reached_queue.push(shared_from_this());
  while (!forward_reached_queue.empty()) {
    auto current_cluster = forward_reached_queue.front();
    forward_reached_queue.pop();
    for (const auto &cluster : current_cluster->out_clusters_) {
      if (cluster->min_ < max_ && cluster->max_ != max_ && forward_reached_clusters.count(cluster) == 0) {
        forward_reached_clusters.insert(cluster);
        forward_reached_queue.push(cluster);
      }
    }
  }
  while (!backward_reached_queue.empty()) {
    auto current_cluster = backward_reached_queue.front();
    backward_reached_queue.pop();
    for (const auto &cluster : current_cluster->in_clusters_) {
      if (cluster->max_ > other->min_ && cluster->max_ != other->max_ &&
          backward_reached_clusters.count(cluster) == 0) {
        backward_reached_clusters.insert(cluster);
        backward_reached_queue.push(cluster);
        if (forward_reached_clusters.count(cluster) != 0) {
          path_clusters.push_back(cluster);
        }
      }
    }
  }
  for (const auto &cluster : path_clusters) {
    Merge(cluster);
  }
  return path_clusters;
}
std::vector<ClusterPtr> Cluster::Inputs() const { return in_clusters_; };
std::vector<ClusterPtr> Cluster::Outputs() const { return out_clusters_; };
std::vector<NodePtr> Cluster::Nodes() const { return nodes_; };

void Cluster::AddFrameInput(InDataAnchorPtr anchor) {
  if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr) {
    inputs_index_[anchor] = inputs_.size();
    inputs_.push_back(anchor);
  }
}

void Cluster::AddFrameOutput(OutDataAnchorPtr anchor) {
  if (anchor != nullptr) {
    outputs_index_[anchor] = outputs_.size();
    outputs_.push_back(anchor);
  }
}

InDataAnchorPtr Cluster::GetFrameInDataAnchor(InDataAnchorPtr anchor) {
  return partition_node_->GetInDataAnchor(static_cast<int>(inputs_index_[anchor]));
}

OutDataAnchorPtr Cluster::GetFrameOutDataAnchor(OutDataAnchorPtr anchor) {
  return partition_node_->GetOutDataAnchor(static_cast<int>(outputs_index_[anchor]));
}

InControlAnchorPtr Cluster::GetFrameInControlAnchor() { return partition_node_->GetInControlAnchor(); };

OutControlAnchorPtr Cluster::GetFrameOutControlAnchor() { return partition_node_->GetOutControlAnchor(); };

Status Cluster::BuildFrame() {
  if (IsUnknownShape() || IsKnownShape() || IsInputNode()) {
    return BuildPartitionFrame();
  } else {
    auto node = nodes_.front();
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          REQUIRE_GRAPH_SUCCESS(
              GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
              "[Remove][Edge] from node %s index %d to node %s failed, index %d.",
              peer_out_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(peer_out_control_anchor),
              in_control_anchor->GetOwnerNode()->GetName().c_str(), AnchorUtils::GetIdx(in_control_anchor));
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    if (IsData() || IsIndependent()) {
      for (const auto &anchor : node->GetAllOutDataAnchors()) {
        AddFrameOutput(anchor);
      }
    } else {
      for (const auto &anchor : node->GetAllInDataAnchors()) {
        AddFrameInput(anchor);
      }
    }
    partition_node_ = node;
  }
  return SUCCESS;
}

Status Cluster::BuildPartitionFrame() {
  auto graph = partitioner_->root_graph_;
  bool is_unknown_shape = IsUnknownShape();
  bool is_input = IsInputNode();
  string known_name = (is_unknown_shape ? "_unknow" : "_know");
  string sub_graph_name_patten = (is_input ? "_input" : known_name);
  std::string sub_graph_name = graph->GetName() + "_sub_" + std::to_string(unique_id_) + sub_graph_name_patten;
  subgraph_ = MakeShared<ComputeGraph>(sub_graph_name);
  REQUIRE_NOT_NULL(subgraph_, "[New][Memory] for subgraph failed, name:%s.", sub_graph_name.c_str());
  auto partition_op = MakeShared<OpDesc>("PartitionedCall_" + std::to_string(unique_id_++), "PartitionedCall");
  REQUIRE_NOT_NULL(partition_op, "[New][Memory] for partition op failed.");
  REQUIRE(AttrUtils::SetBool(partition_op, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
          "[Set][Attr] _is_unknown_shape flag on partitioned op %s failed.", partition_op->GetName().c_str());
  REQUIRE_GRAPH_SUCCESS(partition_op->AddSubgraphName(subgraph_->GetName()),
                        "[Add][SubgraphName] %s for op:%s.",
                        subgraph_->GetName().c_str(), partition_op->GetName().c_str());
  REQUIRE_GRAPH_SUCCESS(partition_op->SetSubgraphInstanceName(0, subgraph_->GetName()),
                        "[Call][SetSubgraphInstanceName] for op:%s failed, index:0, name:%s.",
                        partition_op->GetName().c_str(), subgraph_->GetName().c_str());
  for (auto &node : nodes_) {
    REQUIRE_NOT_NULL(subgraph_->AddNode(node),
                     "[Add][Node] %s to subgraph:%s failed.", node->GetName().c_str(), subgraph_->GetName().c_str());
    REQUIRE(AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
            "[Set][Attr] %s to op:%s failed.", ATTR_NAME_IS_UNKNOWN_SHAPE.c_str(), node->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node),
                          "[Remove][JustNode] failed, graph:%s, node:%s.",
                          graph->GetName().c_str(), node->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(node->SetOwnerComputeGraph(subgraph_),
                          "[Set][OwnerComputeGraph] %s for node:%s failed.",
                          subgraph_->GetName().c_str(), node->GetName().c_str());
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;  // Skip overhang input.
      }
      auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
      if (src_cluster->id_ != id_) {
        AddFrameInput(anchor);
        REQUIRE_GRAPH_SUCCESS(partition_op->AddInputDesc(node->GetOpDesc()->GetInputDesc(anchor->GetIdx())),
                              "[Add][InputDesc] to op:%s failed.", partition_op->GetName().c_str());
      }
    }
    auto in_control_anchor = node->GetInControlAnchor();
    if (in_control_anchor != nullptr) {
      for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
        if (peer_out_control_anchor == nullptr) {
          continue;
        }
        auto src_cluster = partitioner_->node_2_cluster_[peer_out_control_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          REQUIRE_GRAPH_SUCCESS(
              GraphUtils::RemoveEdge(peer_out_control_anchor, in_control_anchor),
              "[Remove][Edge] from %s:%d to %s:%d failed.", peer_out_control_anchor->GetOwnerNode()->GetName().c_str(),
              peer_out_control_anchor->GetIdx(), node->GetName().c_str(), in_control_anchor->GetIdx());
          control_inputs_.insert(src_cluster);
          src_cluster->control_outputs_.insert(peer_out_control_anchor);
        }
      }
    }
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      auto peer_in_anchors = anchor->GetPeerInDataAnchors();
      for (const auto &peer_in_anchor : peer_in_anchors) {
        auto src_cluster = partitioner_->node_2_cluster_[peer_in_anchor->GetOwnerNode()];
        if (src_cluster->id_ != id_) {
          AddFrameOutput(anchor);
          REQUIRE_GRAPH_SUCCESS(partition_op->AddOutputDesc(node->GetOpDesc()->GetOutputDesc(anchor->GetIdx())),
                                "[Add][OutputDesc] to op:%s failed.", partition_op->GetName().c_str());
          break;
        }
      }
    }
  }
  partition_node_ = graph->AddNode(partition_op);
  REQUIRE_NOT_NULL(partition_node_,
                   "[Add][Node] %s to graph:%s failed.", partition_op->GetName().c_str(), graph->GetName().c_str());
  REQUIRE_GRAPH_SUCCESS(partition_node_->SetOwnerComputeGraph(graph),
                        "[Set][OwnerComputeGraph] %s for node:%s failed.",
                        graph->GetName().c_str(), partition_op->GetName().c_str());
  subgraph_->SetParentNode(partition_node_);
  subgraph_->SetParentGraph(graph);
  REQUIRE_GRAPH_SUCCESS(graph->AddSubgraph(subgraph_),
                        "[Add][Subgraph] %s to root graph:%s failed.",
                        subgraph_->GetName().c_str(), graph->GetName().c_str());
  std::string session_graph_id;
  REQUIRE(AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
          "[Get][Attr] %s on root graph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(), graph->GetName().c_str());
  REQUIRE(AttrUtils::SetStr(*subgraph_, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
          "[Set][Attr] %s on subgraph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(), subgraph_->GetName().c_str());
  return SUCCESS;
}

Status Cluster::CombinePartitionFrame() {
  for (const auto &anchor : inputs_) {
    auto peer_out_anchor = anchor->GetPeerOutAnchor();
    auto src_cluster = partitioner_->node_2_cluster_[peer_out_anchor->GetOwnerNode()];
    auto src_anchor = src_cluster->GetFrameOutDataAnchor(peer_out_anchor);
    auto dst_anchor = GetFrameInDataAnchor(anchor);
    REQUIRE_GRAPH_SUCCESS(GraphUtils::RemoveEdge(peer_out_anchor, anchor), "[Remove][Edge] from %s:%d to %s:%d fail.",
                          peer_out_anchor->GetOwnerNode()->GetName().c_str(), peer_out_anchor->GetIdx(),
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx());
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, dst_anchor), "[Add][Edge] from %s:%d to %s:%d failed.",
                          src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                          dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
  }
  for (const auto &src_cluster : control_inputs_) {
    auto src_anchor = src_cluster->GetFrameOutControlAnchor();
    auto dst_anchor = GetFrameInControlAnchor();
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, dst_anchor), "[Add][Edge] from %s:%d to %s:%d failed.",
                          src_anchor->GetOwnerNode()->GetName().c_str(), src_anchor->GetIdx(),
                          dst_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetIdx());
  }
  return SUCCESS;
}

Status Cluster::BuildPartitionSubgraph() {
  if (IsData() || IsNetOutput() || IsIndependent()) {
    return SUCCESS;
  }
  int64_t parent_node_index = 0;
  for (auto anchor : inputs_) {
    auto data_op =
        MakeShared<OpDesc>(subgraph_->GetName() + std::string("Data_") + std::to_string(parent_node_index), ge::DATA);
    REQUIRE_NOT_NULL(data_op, "[New][Memory] for data op failed.");
    auto input_desc = anchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(anchor->GetIdx());
    REQUIRE_GRAPH_SUCCESS(data_op->AddInputDesc(input_desc),
                          "[Add][InputDesc] to op:%s failed.", data_op->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(data_op->AddOutputDesc(input_desc),
                          "[Add][OutputDesc] to op:%s failed.", data_op->GetName().c_str());
    REQUIRE(AttrUtils::SetInt(data_op, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
            "[Set][Attr] %s on subgraph data node:%s failed.",
            ATTR_NAME_PARENT_NODE_INDEX.c_str(), data_op->GetName().c_str());
    bool is_unknown_shape = IsUnknownShape();
    REQUIRE(AttrUtils::SetBool(data_op, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
            "[Set][Attr] %s on data op %s failed.", ATTR_NAME_IS_UNKNOWN_SHAPE.c_str(), data_op->GetName().c_str());
    auto data_node = subgraph_->AddNode(data_op);
    REQUIRE_NOT_NULL(data_node,
                     "[Add][Node] %s to subgraph:%s failed.", data_op->GetName().c_str(), subgraph_->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(data_node->SetOwnerComputeGraph(subgraph_),
                          "[Set][OwnerGraph] %s of data node:%s failed.",
                          subgraph_->GetName().c_str(), data_op->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), anchor),
                          "[Call][AddEdge] Failed add data input edge to %s:%d",
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx());
    parent_node_index++;
  }
  if (outputs_.empty() && control_outputs_.empty()) {
    return SUCCESS;
  }
  auto net_output_op = MakeShared<OpDesc>(subgraph_->GetName() + "_" + NODE_NAME_NET_OUTPUT, ge::NETOUTPUT);
  REQUIRE_NOT_NULL(net_output_op, "[New][Memory] for netoutput op failed.");
  bool is_unknown_shape = IsUnknownShape();
  REQUIRE(AttrUtils::SetBool(net_output_op, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
          "[Set][Attr] %s on op:%s failed.", ATTR_NAME_IS_UNKNOWN_SHAPE.c_str(), net_output_op->GetName().c_str());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    GeTensorDesc input_desc;
    REQUIRE_GRAPH_SUCCESS(net_output_op->AddInputDesc(input_desc),
                          "[Add][InputDesc] to op:%s failed.", net_output_op->GetName().c_str());
  }
  auto net_output_node = subgraph_->AddNode(net_output_op);
  REQUIRE_NOT_NULL(net_output_node,
                   "[Call][AddNode] Failed add netoutput node:%s to subgraph:%s.",
                   net_output_op->GetName().c_str(), subgraph_->GetName().c_str());
  REQUIRE_GRAPH_SUCCESS(net_output_node->SetOwnerComputeGraph(subgraph_),
                        "[Set][OwnerGraph] %s of netoutput node:%s failed.",
                        subgraph_->GetName().c_str(), net_output_node->GetName().c_str());
  parent_node_index = 0;
  for (const auto &anchor : outputs_) {
    auto output_desc = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(anchor->GetIdx()));
    REQUIRE(AttrUtils::SetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index),
            "[Set][Attr] parent_node_index on subgraph node:%s netoutput's input failed.",
            anchor->GetOwnerNode()->GetName().c_str());
    REQUIRE_GRAPH_SUCCESS(net_output_op->UpdateInputDesc(parent_node_index, output_desc),
                          "[Update][InputDesc] of netoutput node:%s failed.", net_output_op->GetName().c_str());

    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInDataAnchor(parent_node_index)),
                          "[Add][Edge] from %s:%d to netoutput node:%s failed.",
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                          net_output_op->GetName().c_str());
    parent_node_index++;
  }
  for (const auto &anchor : control_outputs_) {
    REQUIRE_GRAPH_SUCCESS(GraphUtils::AddEdge(anchor, net_output_node->GetInControlAnchor()),
                          "[Add][ControlEdge] from %s:%d to netoutput node:%s failed.",
                          anchor->GetOwnerNode()->GetName().c_str(), anchor->GetIdx(),
                          net_output_op->GetName().c_str());
  }
  return SUCCESS;
}
void Cluster::Clear() {
  in_clusters_.clear();
  out_clusters_.clear();
  nodes_.clear();
  partitioner_ = nullptr;
  inputs_index_.clear();
  outputs_index_.clear();
  inputs_.clear();
  outputs_.clear();
  control_inputs_.clear();
  control_outputs_.clear();
  partition_node_.reset();
  subgraph_.reset();
  unique_id_ = 0;
}

thread_local size_t Cluster::unique_id_ = 0;
}  // namespace ge
