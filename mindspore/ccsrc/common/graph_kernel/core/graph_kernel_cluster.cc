/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/core/graph_kernel_cluster.h"

#include <algorithm>
#include <set>
#include <utility>
#include <fstream>
#include <string>

#include "utils/hash_map.h"
#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/file_utils.h"
#include "include/common/utils/context/graph_kernel_flags.h"
#include "common/graph_kernel/core/graph_kernel_callback.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {
std::vector<PrimitivePtr> GraphKernelCluster::GetClusterableOpList() {
  std::vector<OpWithLevel> clusterable_ops_with_level = {
    // all target
    {kAllTarget, OpLevel_0, prim::kPrimAbs},
    {kAllTarget, OpLevel_0, prim::kPrimAdd},
    {kAllTarget, OpLevel_0, prim::kPrimCast},
    {kAllTarget, OpLevel_0, prim::kPrimEqual},
    {kAllTarget, OpLevel_0, prim::kPrimExp},
    {kAllTarget, OpLevel_0, prim::kPrimInplaceAssign},
    {kAllTarget, OpLevel_0, prim::kPrimLog},
    {kAllTarget, OpLevel_0, prim::kPrimMaximum},
    {kAllTarget, OpLevel_0, prim::kPrimMinimum},
    {kAllTarget, OpLevel_0, prim::kPrimMul},
    {kAllTarget, OpLevel_0, prim::kPrimNeg},
    {kAllTarget, OpLevel_0, prim::kPrimPow},
    {kAllTarget, OpLevel_0, prim::kPrimRealDiv},
    {kAllTarget, OpLevel_0, prim::kPrimReciprocal},
    {kAllTarget, OpLevel_1, prim::kPrimReduceSum},
    {kAllTarget, OpLevel_1, prim::kPrimReshape},
    {kAllTarget, OpLevel_0, prim::kPrimRound},
    {kAllTarget, OpLevel_0, prim::kPrimRsqrt},
    {kAllTarget, OpLevel_0, prim::kPrimSqrt},
    {kAllTarget, OpLevel_0, prim::kPrimSub},
    {kAllTarget, OpLevel_0, prim::kPrimTanh},
    {kAllTarget, OpLevel_1, prim::kPrimTranspose},
    // ascend
    {kAscendDevice, OpLevel_1, prim::kPrimMatMul},
    {kAscendDevice, OpLevel_1, prim::kPrimTransData},
    {kAscendDevice, OpLevel_1, prim::kPrimBatchMatMul},
    // gpu
    {kGPUDevice, OpLevel_0, prim::kPrimACos},
    {kGPUDevice, OpLevel_0, prim::kPrimAcosh},
    {kGPUDevice, OpLevel_1, prim::kPrimArgMax},
    {kGPUDevice, OpLevel_1, prim::kPrimArgMin},
    {kGPUDevice, OpLevel_0, prim::kPrimAsin},
    {kGPUDevice, OpLevel_0, prim::kPrimAsinh},
    {kGPUDevice, OpLevel_0, prim::kPrimAssign},
    {kGPUDevice, OpLevel_0, prim::kPrimAtan},
    {kGPUDevice, OpLevel_0, prim::kPrimAtan2},
    {kGPUDevice, OpLevel_0, prim::kPrimCos},
    {kGPUDevice, OpLevel_0, prim::kPrimDiv},
    {kGPUDevice, OpLevel_0, prim::kPrimErf},
    {kGPUDevice, OpLevel_0, prim::kPrimExpm1},
    {kGPUDevice, OpLevel_0, prim::kPrimFloor},
    {kGPUDevice, OpLevel_0, prim::kPrimFloorDiv},
    {kGPUDevice, OpLevel_0, prim::kPrimFloorMod},
    {kGPUDevice, OpLevel_0, prim::kPrimGreater},
    {kGPUDevice, OpLevel_0, prim::kPrimGreaterEqual},
    {kGPUDevice, OpLevel_0, prim::kPrimIsFinite},
    {kGPUDevice, OpLevel_0, prim::kPrimIsInf},
    {kGPUDevice, OpLevel_0, prim::kPrimIsNan},
    {kGPUDevice, OpLevel_0, prim::kPrimLess},
    {kGPUDevice, OpLevel_0, prim::kPrimLessEqual},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalOr},
    {kGPUDevice, OpLevel_0, prim::kPrimLogicalNot},
    {kGPUDevice, OpLevel_0, prim::kPrimMod},
    {kGPUDevice, OpLevel_0, prim::kPrimNotEqual},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMax},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMin},
    {kGPUDevice, OpLevel_0, prim::kPrimSelect},
    {kGPUDevice, OpLevel_0, prim::kPrimSign},
    {kGPUDevice, OpLevel_0, prim::kPrimSin},
    {kGPUDevice, OpLevel_0, prim::kPrimStridedSlice},
    {kGPUDevice, OpLevel_1, prim::kPrimCumSum},
    // cpu
    {kCPUDevice, OpLevel_0, prim::kPrimLogicalNot},
    {kCPUDevice, OpLevel_0, prim::kPrimMod},
    {kCPUDevice, OpLevel_1, prim::kPrimReduceMax},
    {kCPUDevice, OpLevel_0, prim::kPrimSelect},
    {kCPUDevice, OpLevel_0, prim::kPrimLess},
    {kCPUDevice, OpLevel_0, prim::kPrimLessEqual},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  auto ops = GkUtils::GetValidOps(clusterable_ops_with_level, flags.fusion_ops_level, flags.enable_cluster_ops_only,
                                  flags.enable_cluster_ops, flags.disable_cluster_ops);
  return GkUtils::FilterExcludedOps(ops);
}

bool GraphKernelCluster::IsClusterableOp(const AnfNodePtr &node) {
  if (AnfUtils::IsGraphKernel(node)) {
    return true;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }

  // For AICPU operators, only the Reshape can be clustered.
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  if (cb->GetTargetFromContext() == kAscendDevice) {
    if (cb->GetProcessor(node) != "aicore" && !IsPrimitiveCNode(node, prim::kPrimReshape)) {
      return false;
    }
  }
  return true;
}

class Graph {
  struct Cluster {
    size_t cluster_id_;        // node_id of the representative.
    size_t cluster_size_{1};   // size of cluster, composite node is considered as one node.
    std::set<size_t> inputs_;  // inputs' cluster_id.
    size_t seed_{0};           // visited flag of dfs.

    Cluster(size_t node_id, const AnfNodePtr &node, const mindspore::HashMap<AnfNodePtr, size_t> &node_idx_map)
        : cluster_id_(node_id) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (const auto &inp : cnode->inputs()) {
        auto iter = node_idx_map.find(inp);
        if (iter != node_idx_map.end()) {
          // At the beginning, cluster_id is equal to node_id
          (void)inputs_.insert(iter->second);
        }
      }
    }
    ~Cluster() = default;

    void Merge(Cluster *other_cluster) {
      other_cluster->cluster_id_ = cluster_id_;
      cluster_size_ += other_cluster->cluster_size_;
      (void)std::for_each(other_cluster->inputs_.begin(), other_cluster->inputs_.end(),
                          [this](size_t inp) { (void)this->inputs_.insert(inp); });
      other_cluster->Clean();
    }

    // clean the info to free memory.
    void Clean() {
      inputs_.clear();
      cluster_size_ = 0;
    }
  };  // struct Cluster

 public:
  // Init and build graph
  Graph(const AnfNodePtrList &nodes, const mindspore::HashMap<AnfNodePtr, size_t> &node_idx_map) {
    clusters_.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
      (void)clusters_.emplace_back(i, nodes[i], node_idx_map);
    }
  }
  ~Graph() = default;

  // find the representative of the cluster
  size_t Find(size_t node_id) {
    size_t &pre_id = clusters_[node_id].cluster_id_;
    return (pre_id == clusters_[pre_id].cluster_id_) ? pre_id : (pre_id = Find(pre_id));
  }

  // merge clusters, the smallest cluster id will be the new cluster id.
  void Merge(const std::vector<size_t> &candidates) {
    size_t min_id = *std::min_element(candidates.begin(), candidates.end());
    for (auto id : candidates) {
      if (id == min_id) continue;
      clusters_[min_id].Merge(&clusters_[id]);
    }
  }

  // Collect nodes together that are in the same cluster.
  std::vector<std::vector<size_t>> CollectClusters() {
    std::vector<std::vector<size_t>> cluster_map(clusters_.size());
    for (size_t i = 0; i < clusters_.size(); i++) {
      cluster_map[Find(i)].push_back(i);
    }
    return cluster_map;
  }

  using VisitFunc = std::function<IncludeType(size_t)>;
  void Dfs(size_t node_id, const VisitFunc &visitor) {
    ++seen_;
    return DepthFirstSearch(Find(node_id), visitor);
  }

  // Get cluster size
  size_t GetSize(size_t cluster_id) { return clusters_[Find(cluster_id)].cluster_size_; }

  // Get cluster's inputs
  const std::set<size_t> &GetInputs(size_t cluster_id) {
    cluster_id = Find(cluster_id);
    RefreshInputs(cluster_id);
    return clusters_[cluster_id].inputs_;
  }

 private:
  void RefreshInputs(size_t i) {
    auto &inputs = clusters_[i].inputs_;
    for (auto iter = inputs.begin(); iter != inputs.end();) {
      size_t new_id = Find(*iter);
      if (new_id != *iter) {
        iter = inputs.erase(iter);
        (void)inputs.insert(new_id);
      } else {
        ++iter;
      }
    }
    (void)inputs.erase(i);
  }

  void DepthFirstSearch(size_t cluster_id, const VisitFunc &visitor) {
    if (clusters_[cluster_id].seed_ >= seen_) return;
    clusters_[cluster_id].seed_ = seen_;
    if (visitor(cluster_id) != FOLLOW) {
      return;
    }
    // traverse inputs in descending order.
    const auto &inputs = GetInputs(cluster_id);
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DepthFirstSearch(*iter, visitor);
    }
  }

  std::vector<Cluster> clusters_;
  size_t seen_{0};
};  // class Graph

class CircleChecker {
 public:
  explicit CircleChecker(const GraphPtr &graph) : graph_(graph) {}
  ~CircleChecker() = default;

  void RemoveCircle(std::vector<size_t> *candidates) {
    if (candidates->size() <= 1) {
      return;
    }
    candidates_.clear();
    candidates_.insert(candidates->begin(), candidates->end());
    for (auto iter = candidates->begin(); iter != candidates->end(); ++iter) {
      if (!candidates_.count(*iter)) continue;
      circle_nodes_.clear();
      if (CheckCircle(*iter)) {
        RemoveCircleNodesFromCandidates();
      }
    }
    (void)candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                           [this](size_t c) { return this->candidates_.count(c) == 0; }),
                            candidates->end());
  }

 private:
  /**
   * Check circle. the candidate is collected into circle_nodes_ if it will form a circle.
   *
   * algorithm:
   * Search from the basenode's input that is NOT in candidates (the basenode is a candidate),
   * If it depends on a node that belongs to candidates, it will form a circle.
   *  e.g.     A -> x -> ... -> B
   *             -> y -> ... -> C
   * In this case, A, B and C are candidates while x and y are not.
   * Both x and y are inputs of A. assumes A is the basenode.
   * When searching from x, the B will be found and added into circle_nodes list,
   * and then when searching from y, the C will be found and added into circle_nodes list.
   */
  bool CheckCircle(size_t basenode) {
    const auto &inputs = graph_->GetInputs(basenode);
    std::set<size_t> visited_circle_nodes;
    for (auto x : inputs) {
      if (candidates_.count(x)) continue;
      bool has_circle = false;
      std::set<size_t> done;
      auto vis_func = [this, &has_circle, &done, &visited_circle_nodes](size_t node_id) {
        if (done.count(node_id) || acyclic_nodes_.count(node_id) || visited_circle_nodes.count(node_id)) {
          return EXCLUDE;
        }
        (void)done.insert(node_id);
        if (candidates_.count(node_id)) {
          has_circle = true;
          circle_nodes_.push_back(node_id);
          return EXCLUDE;
        }
        return FOLLOW;
      };
      graph_->Dfs(x, vis_func);
      if (has_circle) {
        visited_circle_nodes.insert(done.begin(), done.end());
      } else {
        acyclic_nodes_.insert(done.begin(), done.end());
      }
    }
    return !circle_nodes_.empty();
  }

  // remove all circle nodes from candidates
  void RemoveCircleNodesFromCandidates() {
    auto remove_from_candidates = [this](size_t node_id) {
      if (candidates_.count(node_id)) {
        (void)candidates_.erase(node_id);
        return FOLLOW;
      }
      return EXCLUDE;
    };
    for (auto node : circle_nodes_) {
      graph_->Dfs(node, remove_from_candidates);
    }
  }

  GraphPtr graph_;               // bind the global graph
  std::set<size_t> candidates_;  // bind the input candidates
  std::vector<size_t> circle_nodes_;
  std::set<size_t> acyclic_nodes_;
};  // CircleChecker

std::vector<size_t> GraphKernelCluster::FindCandidates(size_t basenode_id) {
  std::vector<size_t> candidates;
  auto include = [this, &candidates, func_graph = nodes_[basenode_id]->func_graph()](size_t cluster_id) {
    const AnfNodePtr &node = this->nodes_[cluster_id];
    if (node->func_graph() != func_graph) {
      return EXCLUDE;
    }
    if (!IsClusterableOp(node) && !IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      return EXCLUDE;
    }
    candidates.push_back(cluster_id);
    // Do not search from clustered node again.
    if (this->graph_->GetSize(cluster_id) > 1) {
      return NOFOLLOW;
    }
    return FOLLOW;
  };
  graph_->Dfs(basenode_id, include);
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

bool GraphKernelCluster::Process(const FuncGraphPtr &func_graph) {
  bool changed = false;
  for (int i = SizeToInt(nodes_.size()) - 1; i >= 0; i--) {
    // if the node has been clustered, it has tried to find its previous nodes, so it's unnecessary to try again.
    if (graph_->GetSize(IntToSize(i)) > 1) {
      continue;
    }
    auto candidates = FindCandidates(IntToSize(i));
    CircleChecker circle_checker(graph_);
    circle_checker.RemoveCircle(&candidates);
    RemoveWildGetitem(&candidates);
    if (candidates.empty()) continue;
    // merge candidates into one cluster
    graph_->Merge(candidates);
  }

  // Rebuild func_graphs
  auto clusters = graph_->CollectClusters();
  for (size_t i = 0; i < clusters.size(); i++) {
    auto node_without_getitem = std::count_if(clusters[i].begin(), clusters[i].end(), [this](size_t node_id) {
      return !IsPrimitiveCNode(this->nodes_[node_id], prim::kPrimTupleGetItem);
    });
    if (node_without_getitem == 0) continue;
    if (node_without_getitem == 1) {
      // Do not cluster a single GraphKernel again.
      // Do not cluster a single Assign.
      const auto &node = nodes_[clusters[i][0]];
      if (AnfUtils::IsGraphKernel(node) || IsPrimitiveCNode(node, prim::kPrimAssign) || !IsClusterableOp(node)) {
        continue;
      }
    }
    CreateFuncGraph(func_graph, clusters[i]);
    changed = true;
  }
  return changed;
}

void GraphKernelCluster::CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id) {
  AnfNodePtrList old_nodes;
  (void)std::transform(nodes_id.begin(), nodes_id.end(), std::back_inserter(old_nodes),
                       [this](size_t id) { return this->nodes_[id]; });
  auto new_node = ReplaceNodesWithGraphKernelNode(old_nodes, func_graph, "fusion");
  if (GraphKernelFlags::GetInstance().dump_as_text) {
    DumpClusterInfo(old_nodes, new_node);
  }
}

void GraphKernelCluster::DumpClusterInfo(const AnfNodePtrList &old_nodes, const AnfNodePtr &new_node) {
  dump_buf_ << "Source nodes of " << new_node->fullname_with_scope() << " = " << new_node->DebugString() << std::endl;
  for (const auto &node : old_nodes) {
    dump_buf_ << "  " << node->fullname_with_scope() << " = " << node->DebugString() << std::endl;
  }
  dump_buf_ << "=======================" << std::endl;
}

void GraphKernelCluster::DumpToFile() {
  auto dir_path = FileUtils::CreateNotExistDirs(std::string("./") + kGraphKernelDumpPath);
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: ./" << kGraphKernelDumpPath;
    return;
  }
  std::string filepath = dir_path.value() + "/" + "graph_kernel_cluster.txt";
  std::ofstream fout(filepath, std::ios::app);
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << filepath << "' failed!";
    return;
  }
  fout << dump_buf_.str() << std::endl;
  fout.close();
}

// The GetItem node should be clustered with its real input.
// If its real input is not in the candidates, the GetItem should be excluded.
void GraphKernelCluster::RemoveWildGetitem(std::vector<size_t> *candidates) {
  bool changed = false;
  std::set<size_t> candidates_set(candidates->begin(), candidates->end());

  for (auto iter = candidates_set.begin(); iter != candidates_set.end();) {
    size_t cluster_id = *iter;
    if (IsPrimitiveCNode(nodes_[cluster_id], prim::kPrimTupleGetItem)) {
      const auto &inputs = graph_->GetInputs(cluster_id);
      if (inputs.size() != 1) {
        MS_LOG(ERROR) << "Input size of GetItem(" << cluster_id << ") should be 1, but got " << inputs.size();
        candidates->clear();
        return;
      }
      auto prev_id = *(inputs.begin());
      if (!candidates_set.count(prev_id)) {
        iter = candidates_set.erase(iter);
        changed = true;
        continue;
      }
    }
    ++iter;
  }
  if (changed) {
    (void)candidates->erase(std::remove_if(candidates->begin(), candidates->end(),
                                           [&candidates_set](size_t c) { return candidates_set.count(c) == 0; }),
                            candidates->end());
  }
}

void GraphKernelCluster::Init(const FuncGraphPtr &func_graph) {
  op_list_ = GetClusterableOpList();
  // process cnode only
  nodes_ = TopoSort(func_graph->get_return(), SuccIncoming,
                    [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  for (size_t i = 0; i < nodes_.size(); i++) {
    node_idx_map_[nodes_[i]] = i;
  }
  graph_ = std::make_shared<Graph>(nodes_, node_idx_map_);
  MS_EXCEPTION_IF_NULL(graph_);
}

bool GraphKernelCluster::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  Init(func_graph);
  bool changed = Process(func_graph);
  if (changed) {
    if (GraphKernelFlags::GetInstance().dump_as_text) {
      DumpToFile();
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  Clean();
  return changed;
}
}  // namespace mindspore::graphkernel
