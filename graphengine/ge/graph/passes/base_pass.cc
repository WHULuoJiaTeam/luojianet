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

#include "graph/passes/base_pass.h"

#include <queue>
#include <unordered_set>

#include "common/debug/log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
constexpr int kMaxRePassTimes = 10000;
constexpr size_t kMaxOneInNodes = 1000;
// Each iteration, we take about 0.3k memory on the stack, we should change the recursion to loop later
constexpr int kMaxRecursiveDepth = 20;

void GetAllNodesNoInputEdge(const ComputeGraphPtr &graph,
                            GEPass::GraphLevelState &g_state) {
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    size_t in_nums = node->GetInNodes().size();
    if (in_nums == 0) {
      g_state.AddNodeToQueueIfNotSeen(node);
    } else if (in_nums > kMaxOneInNodes) {
      g_state.nodes_last.insert(node);
    }
  }
}

bool AnyNodesIn(const Node::Vistor<NodePtr> &nodes, const std::unordered_set<NodePtr> &nodes_set) {
  return std::any_of(nodes.begin(), nodes.end(), [&](const NodePtr &n) {
    return nodes_set.count(n) > 0;
  });
}


bool IsNodeReadyToQueue(const NodePtr &node, GEPass::GraphLevelState &g_state) {
  if (node == nullptr) {
    GELOGW("node is null");
    return false;
  }
  if (g_state.nodes_deleted.count(node) > 0) {
    GELOGD("The node %s was deleted before, skip it.", node->GetName().c_str());
    return false;
  }

  if (g_state.nodes_last.count(node) != 0) {
    return false;
  }

  // all in_node seen && all in_node not suspend
  if (!node->IsAllInNodesSeen(g_state.nodes_seen)) {
    return false;
  }

  if (g_state.nodes_suspend.count(node) > 0) {
    GELOGD("The node %s has been added to suspend-iteration nodes list, the iteration of it will be suspend.",
           node->GetName().c_str());
    return false;
  }

  if (AnyNodesIn(node->GetInAllNodes(), g_state.nodes_suspend)) {
    GELOGD("The node %s has been added to suspend-iteration nodes list, the iteration of it will be suspend.",
           node->GetName().c_str());
    return false;
  }
  return true;
}

void AddNextIterNodes(const NodePtr &cur_node,
                      std::unordered_set<NodePtr> &out_nodes_before_pass,
                      GEPass::GraphLevelState &g_state) {
  for (auto &node : cur_node->GetOutNodes()) {
    if (node == nullptr) {
      continue;
    }
    if(out_nodes_before_pass.erase(node) == 0) {
      // after pass node, new output node come up
      GELOGI("New output node %s come up after pass %s.",
             node->GetName().c_str(), cur_node->GetName().c_str());
    }

    // all in_node seen && all in_node not suspend
    if (IsNodeReadyToQueue(node, g_state)) {
      g_state.AddNodeToQueueIfNotSeen(node);
    }
  }

  //
  for (const auto &node : out_nodes_before_pass) {
    // A-->B-->C  if B was
    // unlink edge may happend, add these node to queue if needed
    if (node->GetInAllNodes().empty() && IsNodeReadyToQueue(node, g_state)) {
      GELOGI("Node %s may lost from cur node, add to queue if not seen.",
             node->GetName().c_str(), cur_node->GetName().c_str());
      g_state.AddNodeToQueueIfNotSeen(node);
    }
  }
}

void AddImmediateRepassNodesToQueue(NodePtr &cur_node,
                                    std::unordered_map<NodePtr, std::string> re_pass_imm_nodes_to_pass_names,
                                    GEPass::GraphLevelState &g_state) {
  for (const auto &node_2_pass_names : re_pass_imm_nodes_to_pass_names) {
    auto imme_repass_node = node_2_pass_names.first;
    if (imme_repass_node == nullptr) {
      GELOGW("Found null immediately re-pass node when executing pass %s on node %s type %s",
             node_2_pass_names.second.c_str(),
             cur_node->GetName().c_str(), cur_node->GetType().c_str());
      continue;
    }
    if (g_state.nodes_passed.count(imme_repass_node) > 0) {
      GELOGD("The node %s specified by pass %s has been passed, it will repass immediately",
             imme_repass_node->GetName().c_str(), node_2_pass_names.second.c_str());
      g_state.AddNodeToQueueFront(imme_repass_node);
      continue;
    }
    GELOGW("The node %s specified by pass %s has un-passed, it will not repass immediately",
           node_2_pass_names.first->GetName().c_str(), node_2_pass_names.second.c_str());
  }
}

void AddLastNodesToQueue(GEPass::GraphLevelState &g_state) {
  for (auto &node : g_state.nodes_last) {
    if (node->IsAllInNodesSeen(g_state.nodes_seen)) {
      g_state.AddNodeToQueueIfNotSeen(node);
    }
  }
  g_state.nodes_last.clear();
}

void AddResumeNodesToQueue(const std::unordered_map<NodePtr, std::string> resume_node_2_pass_names,
                      GEPass::GraphLevelState &g_state) {
  // Now base pass doesnt record the order of suspend & resume, so we dont know which one come first in a node pass.
  // Here if one node pass suspend and resume a node ,consider it resume that node.
  // Better way to record the order, and here suspend or resume in order.
  for (const auto &node_2_pass_names : resume_node_2_pass_names) {
    auto node = node_2_pass_names.first;
    if (g_state.nodes_suspend.erase(node) > 0) {
      if (g_state.nodes_seen.count(node.get()) > 0 || node->IsAllInNodesSeen(g_state.nodes_seen)) {
        g_state.nodes.push_back(node);
        GELOGD("Node %s has been resumed by pass %s, and add to pass queue",
               node->GetName().c_str(), node_2_pass_names.second.c_str());
      }
    }
  }
}

void PushToRePassIfSeen(NodePtr &node, const std::pair<std::string, BaseNodePass *> &name_to_pass,
                        std::unordered_set<Node *> &nodes_seen, const std::vector<NodePtr> &nodes_to_re_pass,
                        GEPass::RepassLevelState &rp_state) {
  for (const auto &node_to_re_pass : nodes_to_re_pass) {
    if (node_to_re_pass == nullptr) {
      GELOGW("Found null re-pass node when executing %s on node %s type %s", name_to_pass.first.c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (nodes_seen.count(node_to_re_pass.get()) > 0 || node_to_re_pass->IsAllInNodesSeen(nodes_seen)) {
      if (rp_state.AddNodeToRepass(node_to_re_pass)) {
        GELOGD("The node %s will be re-pass.", node_to_re_pass->GetName().c_str());
        continue;
      }
      GELOGD("Node %s has been added to repass queue, no need to add again.",  node_to_re_pass->GetName().c_str());
    } else {
      GELOGD("The node %s are not all seen, don't set repass this time", node_to_re_pass->GetName().c_str());
    }
  }
}

void SetFlagOption(NodePassOption option, NamesToPass names_to_pass) {
  for (auto &name_to_pass : names_to_pass) {
    name_to_pass.second->SetOption(option, "");
  }
}

void ClearOption(NamesToPass names_to_pass) {
  for (auto &name_to_pass : names_to_pass) {
    name_to_pass.second->ClearOptions();
  }
}
}  // namespace

Status BaseNodePass::IsolateAndDeleteNode(NodePtr &node, const std::vector<int> &io_map,
                                          bool is_repass_io_immediately) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  GELOGI("Prepare to isolate and delete node, name:%s, type:%s.", node->GetName().c_str(),
         node->GetType().c_str());
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "The owner graph of node:%s must not be null.", node->GetName().c_str());
    GELOGE(FAILED, "[Get][OwnerComputeGraph] failed, The owner graph of node:%s must not be null.",
           node->GetName().c_str());
    return FAILED;
  }

  is_repass_io_immediately ? AddImmediateRePassNodesWithInOut(node) : AddRePassNodesWithInOut(node);

  if (GraphUtils::IsolateNode(node, io_map) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate Node:%s failed", node->GetName().c_str());
    GELOGE(FAILED, "[Isolate][Node] %s failed.", node->GetName().c_str());
    return FAILED;
  }

  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "call RemoveNodeWithoutRelink for node:%s failed.", node->GetName().c_str());
    GELOGE(FAILED, "[Call][RemoveNodeWithoutRelink] for node:%s failed.", node->GetName().c_str());
    return FAILED;
  }

  AddNodeDeleted(node);
  return SUCCESS;
}

Status GEPass::Run(const NamesToPass &names_to_passes) {
  if (graph_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph_ is nullptr, check invalid.");
    GELOGE(INTERNAL_ERROR, "[Check][Param] The graph is nullptr");
    return INTERNAL_ERROR;
  }
  if (names_to_passes.empty()) {
    GELOGW("No passes input, the GEPass will do nothing");
    return INTERNAL_ERROR;
  }
  for (const auto &name_to_pass : names_to_passes) {
    if (name_to_pass.second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] There is null pointer in passes(%s)", name_to_pass.first.c_str());
      return INTERNAL_ERROR;
    }
  }

  if (depth_ > kMaxRecursiveDepth) {
    GELOGE(PARAM_INVALID,
        "[Check][Param] The pass for root graph %s will be terminated because too many nesting"
        " levels(%d) of subgraphs, last subgraph is %s",
        root_graph_->GetName().c_str(), depth_, graph_->GetName().c_str());
    return PARAM_INVALID;
  }

  return RunPassesOneGraph(names_to_passes);
  // todo debug mode is on, find first node in topo order which is not passed. and give a warning
}

void NotifyPassGraphStart(const ComputeGraphPtr &graph, const NamesToPass &names_to_pass) {
  for (auto &name_to_pass : names_to_pass) {
    name_to_pass.second->OnStartPassGraph(graph);
  }
}

Status GEPass::HandleLeakedSuspendNodes(const NamesToPass &names_to_passes, GraphLevelState &g_state) {
  std::unordered_map<NodePtr, std::string> resume_nodes_to_pass_names;
  for (auto &name_to_pass : names_to_passes) {
    name_to_pass.second->init();
    auto ret = name_to_pass.second->OnSuspendNodesLeaked();
    if (ret != SUCCESS) {
      GELOGE(ret, "Internal error with OnSuspendNodesLeaked on pass %s.", name_to_pass.first.c_str());
      return ret;
    }
    for (const auto &resume_node : name_to_pass.second->GetNodesResume()){
      resume_nodes_to_pass_names[resume_node].append(name_to_pass.first + ",");
    }
  }
  AddResumeNodesToQueue(resume_nodes_to_pass_names, g_state);
  return SUCCESS;
}

Status GEPass::RunPassesOneGraph(const NamesToPass &names_to_passes) {
  GELOGD("Begin to run pass on graph, passes count %zu", names_to_passes.size());
  NotifyPassGraphStart(graph_, names_to_passes);
  GraphLevelState g_state;
  g_state.re_pass_times = 0;
  GetAllNodesNoInputEdge(graph_, g_state);
  GELOGD("Start points count %zu", g_state.nodes.size());

  do {
    if (!g_state.nodes_suspend.empty()) {
      auto ret = HandleLeakedSuspendNodes(names_to_passes, g_state);
      if (ret != SUCCESS) {
        // log inside upper function
        return ret;
      }
      if (g_state.nodes.empty()) {
        GELOGE(INTERNAL_ERROR, "There are some suspended nodes leaked and no pass resume them.");
        return INTERNAL_ERROR;
      }
    }
    auto ret = RunPassesGraphRepass(names_to_passes, g_state);
    if (ret != SUCCESS) {
      return ret;
    }
  } while (!g_state.nodes_suspend.empty());

  return SUCCESS;
}


Status GEPass::RunPassesGraphRepass(const NamesToPass &names_to_passes, GraphLevelState &g_state) {
  RepassLevelState rp_state;
  do {
    for (auto &node : rp_state.nodes_re_pass) {
      if (rp_state.nodes_re_pass_set.count(node) > 0) {
        GELOGD("Add node %s to queue for re-pass", node->GetName().c_str());
        g_state.AddNodeToQueue(node);
      }
    }
    rp_state.ClearRepass();

    while (!g_state.nodes.empty()) {
      auto node = g_state.PopFront();
      if (g_state.nodes_deleted.count(node) > 0) {
        GELOGD("The node %s was deleted before, skip it.", node->GetName().c_str());
        continue;
      }
      rp_state.EraseNodeFromRepass(node);
      g_state.nodes_seen.insert(node.get());

      // collect out nodes before pass
      std::unordered_set<NodePtr> out_nodes_before_pass;
      for (const auto &out_node : node->GetOutNodes()) {
        out_nodes_before_pass.insert(out_node);
      }
      auto ret = RunPassesNodeOnce(node, names_to_passes, g_state, rp_state);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Process][Passes] on node %s type %s failed, error code:%u", node->GetName().c_str(),
               node->GetType().c_str(), ret);
        return ret;
      }
      AddNextIterNodes(node, out_nodes_before_pass, g_state);

    }
    AddLastNodesToQueue(g_state);
  } while ((!rp_state.nodes_re_pass.empty() || !g_state.nodes.empty()) && ++g_state.re_pass_times < kMaxRePassTimes);

  if (g_state.re_pass_times == kMaxRePassTimes) {
    GELOGW("re_pass_times should not come to %d", kMaxRePassTimes);
  }
  GELOGD("All passes runs end");
  return SUCCESS;
}

Status GEPass::RunPassesOnSubGraph(const NodePtr &node, const NamesToPass &names_to_passes, bool &has_sub_graph) {
  auto sub_graph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
  has_sub_graph = false;
  for (const auto &name : sub_graph_names) {
    auto graph = root_graph_->GetSubgraph(name);
    if (graph == nullptr) {
      GELOGW("Can not find the sub graph %s from node %s, the pass-process will skip it",
          name.c_str(), node->GetName().c_str());
      continue;
    }
    has_sub_graph = true;
    GELOGI("Begin to run passes on the sub graph %s of node %s", name.c_str(), node->GetName().c_str());
    GEPass pass(graph, root_graph_, depth_ + 1);
    auto ret = pass.Run(names_to_passes);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][Passes] for sub graph:%s from node:%s failed", name.c_str(), node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status GEPass::RunPassesNodeOnce(NodePtr &node, const NamesToPass &names_to_passes,
                                 GraphLevelState &g_state, RepassLevelState &rp_state) {
  auto ret = RunPassesOnNode(node, names_to_passes, g_state, rp_state);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Process][Passes] on node %s type %s failed, error code:%u", node->GetName().c_str(),
           node->GetType().c_str(), ret);
    return ret;
  }

  bool has_sub_graph = false;
  ret = RunPassesOnSubGraph(node, names_to_passes, has_sub_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][Passes] on the sub graph of node %s failed", node->GetName().c_str());
    return ret;
  }

  if (has_sub_graph) {
    GELOGD("There are subgraphs on node %s, run passes for for the second time", node->GetName().c_str());
    SetFlagOption(kOptimizeAfterSubGraph, names_to_passes);
    ret = RunPassesOnNode(node, names_to_passes, g_state, rp_state);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Process][Passes] on node %s type %s failed, error code: %u", node->GetName().c_str(),
             node->GetType().c_str(), ret);
      return ret;
    }

    // There is only one option scene, so set and clear options around the `RunPasses` func.
    // if there are more than one scene to set options, the `ClearOption` function
    // should be called each time at the begin of the iteration
    ClearOption(names_to_passes);
  }
  return SUCCESS;
}

Status GEPass::RunPassesOnNode(NodePtr &node, const NamesToPass &names_to_passes, GraphLevelState &g_state,
                               RepassLevelState &rp_state) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  GELOGD("Begin to run pass for node %s", node->GetName().c_str());
  for (const auto &name_to_pass : names_to_passes) {
    GELOGD("Begin to run pass %s for node %s", name_to_pass.first.c_str(), node->GetName().c_str());
    name_to_pass.second->init();
    auto result = name_to_pass.second->Run(node);
    if (result != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "process pass %s on node:%s failed, ret:%u",
                        name_to_pass.first.c_str(), node->GetName().c_str(), result);
      GELOGE(INTERNAL_ERROR, "[Process][Pass] %s on node %s failed, result "
                             "%u, the passes will be terminated immediately.",
             name_to_pass.first.c_str(), node->GetName().c_str(), result);
      return result;
    }
    if (name_to_pass.second->GetNodesDeleted().count(node) > 0) {
      GELOGD("The node %s was deleted by pass %s, stop the remain passes", node->GetName().c_str(),
             name_to_pass.first.c_str());
      break;
    }
  }

  g_state.nodes_passed.insert(node);

  std::unordered_map<NodePtr, std::string> re_pass_imm_nodes_to_pass_names;
  std::unordered_map<NodePtr, std::string> resume_nodes_to_pass_names;
  // if muti psss repass one same node, it will add to queue many times, so collect and duplicate
  for (const auto &name_to_pass : names_to_passes) {
    PushToRePassIfSeen(node, name_to_pass, g_state.nodes_seen,
                       name_to_pass.second->GetNodesNeedRePass(),
                       rp_state);
    // collect imm_node && resume_node among these passes
    for (const auto &imm_node : name_to_pass.second->GetNodesNeedRePassImmediately()){
      re_pass_imm_nodes_to_pass_names[imm_node].append(name_to_pass.first + ",");
    }
    for (const auto &resume_node : name_to_pass.second->GetNodesResume()){
      resume_nodes_to_pass_names[resume_node].append(name_to_pass.first + ",");
    }

    for (const auto &suspend_node : name_to_pass.second->GetNodesSuspend()) {
      GELOGD("The iteration suspend of node %s has been set by pass %s", suspend_node->GetName().c_str(),
             name_to_pass.first.c_str());
      g_state.nodes_suspend.insert(suspend_node);
    }
    const auto &nodes_deleted_by_pass = name_to_pass.second->GetNodesDeleted();
    g_state.nodes_deleted.insert(nodes_deleted_by_pass.begin(), nodes_deleted_by_pass.end());
  }

  AddImmediateRepassNodesToQueue(node, re_pass_imm_nodes_to_pass_names, g_state);
  AddResumeNodesToQueue(resume_nodes_to_pass_names, g_state);

  return SUCCESS;
}
}  // namespace ge
