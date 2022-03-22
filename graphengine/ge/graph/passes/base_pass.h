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

#ifndef GE_GRAPH_PASSES_BASE_PASS_H_
#define GE_GRAPH_PASSES_BASE_PASS_H_

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
enum NodePassOption {
  // if there is a sub graph on the node, the pass on the node will do:
  // Pass(node) -> pass all sub graphs on the node -> Pass(node)
  // when pass the node for the second time, the kOptimizeAfterSubGraph will be set as a flag key
  kOptimizeAfterSubGraph,

  // add new options before kOptionEnd
  kOptionEnd
};

class BaseNodePass {
  // todo comments
 public:
  ///
  /// Optimize on one node. the function can add nodes to the graph, change
  /// connections between nodes while optimizing or remove nodes from the graph.
  /// @param node
  /// @return
  ///
  virtual Status Run(NodePtr &node) = 0;

  virtual ~BaseNodePass() = default;

  const std::vector<NodePtr> &GetNodesNeedRePass() { return nodes_need_re_pass_; }

  const std::unordered_set<NodePtr> &GetNodesNeedRePassImmediately() { return nodes_need_re_pass_immediately_; }

  const std::unordered_set<NodePtr> &GetNodesDeleted() { return nodes_deleted_; }

  const std::unordered_set<NodePtr> &GetNodesSuspend() { return nodes_suspend_; }

  const std::unordered_set<NodePtr> &GetNodesResume() { return nodes_resume_; }

  virtual Status OnSuspendNodesLeaked() { return SUCCESS; }

  void SetOption(NodePassOption option, const std::string &value) { options_[option] = value; }

  void ClearOptions() { options_.clear(); }

  void init() {
    nodes_need_re_pass_.clear();
    nodes_need_re_pass_immediately_.clear();
    nodes_deleted_.clear();
    nodes_suspend_.clear();
    nodes_resume_.clear();
  }

  virtual void OnStartPassGraph(const ComputeGraphPtr &graph) {
    current_graph_name_ = graph->GetName();
  }

 protected:
  const string &GetCurrentGraphName() const {
    return current_graph_name_;
  }
  Status IsolateAndDeleteNode(NodePtr &node, const std::vector<int> &io_map, bool is_repass_io_immediately = false);

  Status IsolateAndDeleteNode(NodePtr &node, const std::initializer_list<int> &io_map, bool is_repass_io_immediately = false) {
    return IsolateAndDeleteNode(node, std::vector<int>(io_map), is_repass_io_immediately);
  }

  ///
  /// Add a node to be optimized again. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  ///
  void AddRePassNode(const NodePtr &node) { nodes_need_re_pass_.emplace_back(node); }

  ///
  /// Add a node to be optimized immediately again. If you add a new node to the graph, or
  /// change a node connections, and you want to make sure the node will be
  /// optimized by other passes, call this function.
  /// @param node
  ///
  void AddImmediateRePassNode(const NodePtr &node) { nodes_need_re_pass_immediately_.insert(node); }

  ///
  /// Add a node and it's input/output data nodes to be optimized again.
  /// @param node
  ///
  void AddRePassNodesWithInOut(const NodePtr &node) {
    auto in_nodes = node->GetInNodes();
    for (auto &in_node : in_nodes) {
      AddRePassNode(in_node);
    }
    AddRePassNode(node);
    auto out_nodes = node->GetOutNodes();
    for (auto &out_node : out_nodes) {
      AddRePassNode(out_node);
    }
  }

  ///
  /// Add a node and it's input/output data nodes to be optimized immediately again.
  /// @param node
  ///
  void AddImmediateRePassNodesWithInOut(const NodePtr &node) {
    auto in_nodes = node->GetInNodes();
    for (auto &in_node : in_nodes) {
      AddImmediateRePassNode(in_node);
    }
    AddImmediateRePassNode(node);
    auto out_nodes = node->GetOutNodes();
    for (auto &out_node : out_nodes) {
      AddImmediateRePassNode(out_node);
    }
  }

  ///
  /// If you deleted a node from the graph, especially current node. The remain
  /// iterate passes will continue process on the deleted node(if it can be
  /// reached by edge connections) till the last one. Obviously it is a waste of
  /// time. You can add the deleted nodes by calling this function, to stop the
  /// next iterations.
  /// @param node
  ///
  void AddNodeDeleted(const NodePtr &node) { nodes_deleted_.insert(node); }

  ///
  /// If you postpone a node from the graph, especially following node. The remain
  /// iterate passes will stop process on the postpone node(if it can be
  /// reached by edge connections) till the last one. Obviously it is a waste of
  /// time. You can add the postpone nodes by calling this function, to stop the
  /// next iterations.
  /// @param node
  ///
  void AddNodeSuspend(const NodePtr &node) { nodes_suspend_.insert(node); }

  void AddNodeResume(const NodePtr &node) { nodes_resume_.insert(node); }

  bool OptionExists(NodePassOption option) { return options_.count(option) > 0; }

 private:
  std::vector<NodePtr> nodes_need_re_pass_;
  std::unordered_set<NodePtr> nodes_need_re_pass_immediately_;
  std::unordered_set<NodePtr> nodes_deleted_;
  std::unordered_set<NodePtr> nodes_suspend_;
  std::unordered_set<NodePtr> nodes_resume_;
  std::map<NodePassOption, std::string> options_;
  std::string current_graph_name_;
};

using NamesToPass = std::vector<std::pair<std::string, BaseNodePass *>>;

class GEPass {
 public:
  explicit GEPass(ComputeGraphPtr &graph) : graph_(graph), root_graph_(graph), depth_(1) {}
  virtual ~GEPass() = default;
  Status Run(const NamesToPass &names_to_passes);
  /*
* todo
* OneGraph: nodes_deleted, nodes_seen, nodes_passed, nodes_suspended
* RePass: nodes_re_pass
* GraphOneTime: nodes_last
* NodeOneTime: nodes_re_pass_immediately, nodes_resume
*/
  struct GraphLevelState {
    std::unordered_set<NodePtr> nodes_deleted;
    std::unordered_set<Node *> nodes_seen;
    std::unordered_set<NodePtr> nodes_passed;
    std::unordered_set<NodePtr> nodes_suspend;
    std::unordered_set<NodePtr> nodes_last;
    std::deque<NodePtr> nodes;
    int re_pass_times;

    void AddNodeToQueueFront(NodePtr node) {
      nodes_seen.insert(node.get());
      nodes.emplace_front(std::move(node));
    }

    void AddNodeToQueue(NodePtr node) {
      nodes_seen.insert(node.get());
      nodes.emplace_back(std::move(node));
    }
    void AddNodeToQueueIfNotSeen(NodePtr node) {
      if (nodes_seen.insert(node.get()).second) {
        nodes.emplace_back(std::move(node));
      }
    }
    NodePtr PopFront() {
      NodePtr node = nodes.front();
      nodes.pop_front();
      return node;
    }
  };
  struct RepassLevelState {
    std::vector<NodePtr> nodes_re_pass;
    std::unordered_set<NodePtr> nodes_re_pass_set;
    bool AddNodeToRepass(NodePtr node) {
      if (!nodes_re_pass_set.insert(node).second) {
        return false;
      }
      nodes_re_pass.emplace_back(node);
      return true;
    }
    void EraseNodeFromRepass(NodePtr node) {
      nodes_re_pass_set.erase(node);
    }
    void ClearRepass() {
      nodes_re_pass_set.clear();
      nodes_re_pass.clear();
    }
  };
  struct GraphOneTimeLevelState {
    std::unordered_set<NodePtr> nodes_last;
  };

 private:
  GEPass(ComputeGraphPtr &graph, ComputeGraphPtr &root_graph, int depth)
      : graph_(graph), root_graph_(root_graph), depth_(depth) {}
  Status RunPassesNodeOnce(NodePtr &node, const NamesToPass &names_to_passes,
                           GraphLevelState &g_state, RepassLevelState &rp_state);
  Status RunPassesGraphRepass(const NamesToPass &names_to_passes, GraphLevelState &g_state);
  Status RunPassesOneGraph(const NamesToPass &names_to_passes);
  Status RunPassesOnSubGraph(const NodePtr &node, const NamesToPass &names_to_passes, bool &has_sub_graph);
  Status RunPassesOnNode(NodePtr &node, const NamesToPass &names_to_passes, GraphLevelState &g_state,
                         RepassLevelState &rp_state);
  Status HandleLeakedSuspendNodes(const NamesToPass &names_to_passes, GraphLevelState &g_state);
  ComputeGraphPtr graph_;
  ComputeGraphPtr root_graph_;
  int depth_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_BASE_PASS_H_
