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

#include "graph/compute_graph.h"

#include <deque>
#include "graph/format_refiner.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "ge/ge_api_types.h"
#include "graph/shape_refiner.h"
#include "graph/compute_graph_impl.h"
#include "proto/ge_ir.pb.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/common/string_util.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {
const size_t OUTPUT_PARAM_SIZE = 2UL;
bool IsUseBFS() {
  std::string run_mode;
  const int32_t base = 10;
  if ((ge::GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == GRAPH_SUCCESS) && (!run_mode.empty())) {
    if (static_cast<GraphRunMode>(std::strtol(run_mode.c_str(), nullptr, base)) >= TRAIN) {
      return true;
    }
  } else {
    GELOGI("OPTION_GRAPH_RUN_MODE not set, use BFSTopologicalSorting by default.");
  }
  return false;
}
}  // namespace

ComputeGraphImpl::ComputeGraphImpl(const std::string &name)
    : name_(name),
      nodes_(),
      input_nodes_(),
      sub_graph_(),
      is_valid_flag_(false),
      need_iteration_(false) {
}

std::string ComputeGraphImpl::GetName() const { return name_; }

void ComputeGraphImpl::SetName(const std::string &name) { name_ = name; }

size_t ComputeGraphImpl::GetAllNodesSize(const ConstComputeGraphPtr &compute_graph) const {
  return GetAllNodes(compute_graph).size();
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetAllNodes(const ConstComputeGraphPtr &compute_graph) const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodes(subgraphs, compute_graph);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetAllNodes(const NodeFilter &node_filter,
                                                                const GraphFilter &graph_filter,
                                                                const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;

  (void)candidates.insert(candidates.begin(), nodes_.begin(), nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    candidates.pop_front();

    if (node_filter == nullptr || node_filter(*node)) {
      all_nodes.emplace_back(node);
    }

    const OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
        const auto subgraph = GetSubgraph(*name_iter);
      if (subgraph == nullptr) {
        continue;
      }
      if ((graph_filter == nullptr) || graph_filter(*node, name_iter->c_str(), subgraph)) {
        auto subgraph_nodes = subgraph->GetDirectNode();
        (void)(candidates.insert(candidates.begin(), subgraph_nodes.begin(), subgraph_nodes.end()));
      }
    }
  }

  return Vistor<NodePtr>(compute_graph, all_nodes);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs,
                                                                  const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;

  (void)candidates.insert(candidates.begin(), nodes_.begin(), nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    const OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto subgraph = GetSubgraph(*name_iter);
      if (subgraph != nullptr) {
        subgraphs.emplace_back(subgraph);
        auto subgraph_nodes = subgraph->GetDirectNode();
          (void)candidates.insert(candidates.begin(), subgraph_nodes.begin(), subgraph_nodes.end());
      }
    }
  }

  return Vistor<NodePtr>(compute_graph, all_nodes);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetNodes(const bool is_unknown_shape,
                                                             const ConstComputeGraphPtr &compute_graph) const {
  if (is_unknown_shape) {
    return GetDirectNode(compute_graph);
  } else {
    return GetAllNodes(compute_graph);
  }
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetNodes(const bool is_unknown_shape,
                                                             const NodeFilter &node_filter,
                                                             const GraphFilter &graph_filter,
                                                             const ConstComputeGraphPtr &compute_graph) const {
  return is_unknown_shape ? GetDirectNode(compute_graph) : GetAllNodes(node_filter, graph_filter, compute_graph);
}

size_t ComputeGraphImpl::GetDirectNodesSize() const { return direct_nodes_size_; }

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetDirectNode(const ConstComputeGraphPtr &compute_graph) const {
  return Vistor<NodePtr>(compute_graph, nodes_);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetInputNodes(const ConstComputeGraphPtr &compute_graph) const {
  return Vistor<NodePtr>(compute_graph, input_nodes_);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetOutputNodes(const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> result;
  for (auto iter = output_nodes_info_.begin(); iter != output_nodes_info_.end(); ++iter) {
    result.push_back(iter->first);
  }
  return Vistor<NodePtr>(compute_graph, result);
}

NodePtr ComputeGraphImpl::FindNode(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetName() == name) {
      return node;
    }
  }
  return nullptr;
}

NodePtr ComputeGraphImpl::FindFirstNodeMatchType(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() == name) {
      return node;
    }
  }
  return nullptr;
}

bool ComputeGraphImpl::GraphAttrsAreEqual(const ComputeGraphImpl &r_graph) const {
  // 整改前实现中，只比较了属性名字，没有比较属性内容，暂时维持这个玩法
  return attrs_.GetAllAttrNames() == r_graph.attrs_.GetAllAttrNames();
}

/// Since there may be different input nodes
/// chosen by user in the same graph, special judgment is needed
bool ComputeGraphImpl::VectorInputNodePtrIsEqual(const std::vector<NodePtr> &left_nodes,
                                                 const std::vector<NodePtr> &right_nodes) const {
  const auto left_nodes_size = left_nodes.size();
  const auto right_nodes_size = right_nodes.size();
  if (left_nodes_size != right_nodes_size) {
    REPORT_INNER_ERROR("E19999", "Check failed with graph input_nodes_: "
                       "left inputNodes size %zu is different with right inputNodes size %zu .",
                       left_nodes_size, right_nodes_size);
    GELOGE(GRAPH_FAILED, "[Check][Param] failed with graph input_nodes_: "
           "left inputNodes size %zu is different with right inputNodes size %zu .",
           left_nodes_size, right_nodes_size);
    return false;
  }
  for (size_t j = 0UL; j < left_nodes_size; j++) {
    if (left_nodes.at(j) == nullptr || right_nodes.at(j) == nullptr) {
      REPORT_INNER_ERROR("E19999", "left_nodes.at(%zu) or right_nodes.at(%zu) is nullptr", j, j);
      GELOGE(GRAPH_FAILED, "[Check][Param] left_nodes.at(%zu) or right_nodes.at(%zu) is nullptr", j, j);
      return false;
    }
    const auto &left_input_name = left_nodes.at(j)->GetName();
    const auto &right_input_name = right_nodes.at(j)->GetName();
    if (left_input_name != right_input_name) {
      REPORT_INNER_ERROR("E19999", "Check failed with graph input_nodes_: "
                         "left inputNode name %s is different with right inputNode name %s at inputNodes index %zu.",
                         left_input_name.c_str(), right_input_name.c_str(), j);
      GELOGE(GRAPH_FAILED, "[Check][Param] failed with graph input_nodes_: "
             "left inputNode name %s is different with right inputNode name %s at inputNodes index %zu.",
             left_input_name.c_str(), right_input_name.c_str(), j);
      return false;
    }
  }
  return true;
}

bool ComputeGraphImpl::GraphMembersAreEqual(const ComputeGraphImpl &r_graph) const {
  return (IsEqual(this->sub_graph_.size(), r_graph.sub_graph_.size(), "graph.subgraphs_.size()") &&
          IsEqual(this->GetDirectNodesSize(), r_graph.GetDirectNodesSize(), "graph.nodes_.size()") &&
          VectorInputNodePtrIsEqual(this->input_nodes_, r_graph.input_nodes_) &&
          IsEqual(this->name_, r_graph.name_, "graph.name_") &&
          IsEqual(this->is_valid_flag_, r_graph.is_valid_flag_, "graph.is_valid_flag_") &&
          IsEqual(this->need_iteration_, r_graph.need_iteration_, "graph.need_iteration_") &&
          IsEqual(this->params_share_map_, r_graph.params_share_map_, "graph.params_share_map_") &&
          IsEqual(this->out_nodes_map_, r_graph.out_nodes_map_, "graph.out_nodes_map_") &&
          IsEqual(this->inputs_order_, r_graph.inputs_order_, "graph.inputs_order_") &&
          IsEqual(this->output_size_, r_graph.output_size_, "graph.output_size_") &&
          IsEqual(this->input_size_, r_graph.input_size_, "graph.input_size_") &&
          IsEqual(this->output_nodes_info_, r_graph.output_nodes_info_, "graph.output_nodes_info_"));
}

bool ComputeGraphImpl::operator==(const ComputeGraphImpl &r_graph) const {
  // Firstly: Graph's members equal
  if ((!GraphMembersAreEqual(r_graph)) || (!GraphAttrsAreEqual(r_graph))) {
    return false;
  }

  // Secondly: Node equal means the link relationship between node and node itself equal
  for (const auto &left_node : nodes_) {
    if (left_node == nullptr) {
      REPORT_INNER_ERROR("E19999", "left_node is nullptr, graph:%s", this->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] left_node is nullptr");
      return false;
    }
    const auto &node_name = left_node->GetName();
    // After TopologicalSorting, node order can change, so find node by name
    const auto &right_node = r_graph.FindNode(node_name);
    GE_IF_BOOL_EXEC(right_node == nullptr,
                    REPORT_INNER_ERROR("E19999", "left_node:%s not find in r_graph:%s",
                                       node_name.c_str(), r_graph.GetName().c_str());
                    GELOGE(GRAPH_FAILED, "[Check][Param] right_node is NULL!!!"); return false);
    if (!(*right_node == *left_node)) {
      REPORT_INNER_ERROR("E19999", "Compare graph failed, node:%s not equal.", node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Compare][Graph] failed, node:%s not equal.", node_name.c_str());
      return false;
    }
  }

  // Thirdly: Recursively determine whether the sub graphs are equal
  for (size_t i = 0UL; i < this->sub_graph_.size(); i++) {
    if (!(*((this->sub_graph_)[i]) == *((r_graph.sub_graph_)[i]))) {
      return false;
    }
  }
  return true;
}

NodePtr ComputeGraphImpl::AddNodeFront(const NodePtr node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr or op desc should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDesc()->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  if ((GetDirectNodesSize() > 0UL) && ((*(nodes_.begin()))->GetType() == DATA)) {
    InsertToNodeList(next(nodes_.begin()), node);
  } else {
    InsertToNodeList(nodes_.begin(), node);
  }
  return node;
}

NodePtr ComputeGraphImpl::AddNodeFront(const OpDescPtr &op,
                                       const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERROR("E19999", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "node %s init failed.", op->GetName().c_str());
                  GELOGE(GRAPH_FAILED, "node init fail."); return nullptr);
  return AddNodeFront(node_ptr);
}

NodePtr ComputeGraphImpl::AddNode(NodePtr node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "the node ptr or op desc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc ptr should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDesc()->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  PushBackToNodeList(node);
  return node;
}

NodePtr ComputeGraphImpl::AddNode(OpDescPtr op, const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERROR("E19999", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node_ptr == nullptr,
                  REPORT_CALL_ERROR("E19999", "create node failed.");
                  GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "node:%s init failed.", op->GetName().c_str());
                  GELOGE(GRAPH_FAILED, "[Init][Node] %s fail.", op->GetName().c_str()); return nullptr);
  return AddNode(node_ptr);
}

NodePtr ComputeGraphImpl::AddNode(OpDescPtr op, const int64_t id, const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERROR("E19999", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(id);
  NodePtr node = shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node == nullptr,
                  REPORT_CALL_ERROR("E19999", "create node failed.");
                  GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node->Init() != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "node init failed.");
                  GELOGE(GRAPH_FAILED, "[Init][Node] fail."); return nullptr);
  node->SetHostNode(is_valid_flag_);
  PushBackToNodeList(node);
  return node;
}

NodePtr ComputeGraphImpl::AddInputNode(const NodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return nullptr;
  }
  input_nodes_.push_back(node);
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "[Add][Node] failed");
  }
  return node;
}

NodePtr ComputeGraphImpl::AddOutputNode(const NodePtr node) {
  return AddOutputNodeByIndex(node, 0);
}

NodePtr ComputeGraphImpl::AddOutputNodeByIndex(const NodePtr node, const int32_t index) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr or opdesc should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or opdesc should not be null.");
    return nullptr;
  }

  bool already_have = false;
  NodePtr result = node;
  // [output_nodes_info_ : should not be null]
  for (const auto &item : output_nodes_info_) {
    if ((item.first->GetName() == node->GetName()) && (item.second == index)) {
      already_have = true;
      result = item.first;
      break;
    }
  }

  if (!already_have) {
    output_nodes_info_.emplace_back(std::make_pair(node, index));
    GELOGI("Push back node name:%s, index:%d, into output_nodes_info_.", node->GetName().c_str(), index);
  }

  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "[Add][Node] failed");
  }
  return result;
}

graphStatus ComputeGraphImpl::RemoveConstInput(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr || out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    if (out_anchor->GetOwnerNode()->GetType() == CONSTANT || out_anchor->GetOwnerNode()->GetType() == CONSTANTOP) {
      GE_CHK_BOOL_RET_STATUS(GraphUtils::RemoveEdge(out_anchor, in_anchor) == GRAPH_SUCCESS, GRAPH_FAILED,
                             "[Remove][Edge] from const op %s failed.", out_anchor->GetOwnerNode()->GetName().c_str());
      if (out_anchor->GetOwnerNode()->GetOutNodes().size() == 0UL) {
        GELOGI("Remove const op %s.", out_anchor->GetOwnerNode()->GetName().c_str());
        const auto iter = find(nodes_.begin(), nodes_.end(), out_anchor->GetOwnerNode());
        if (iter != nodes_.end()) {
          EraseFromNodeList(iter);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::RemoveNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // delete const op for this node
  (void)RemoveConstInput(node);

  // if the node save as input node, delete it
  (void)RemoveInputNode(node);

  // if the node save as input node, delete it
  (void)RemoveOutputNode(node);

  if (GRAPH_SUCCESS != IsolateNode(node)) {
    GELOGE(GRAPH_FAILED, "[Isolate][Node] failed, node name: %s, graph:%s.", node->GetName().c_str(),
           name_.c_str());
    return GRAPH_FAILED;
  }

  const auto iter = find(nodes_.begin(), nodes_.end(), node);
  if (iter != nodes_.end()) {
    EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraphImpl::RemoveInputNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  const auto iter = find(input_nodes_.begin(), input_nodes_.end(), node);
  if (iter != input_nodes_.end()) {
    (void)input_nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraphImpl::RemoveOutputNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  auto iter = output_nodes_info_.begin();
  bool find_node = false;
  // [output_nodes_info_ : should not be null]
  while (iter != output_nodes_info_.end()) {
    if (node->GetName() == iter->first->GetName()) {
      iter = output_nodes_info_.erase(iter);
      find_node = true;
    } else {
      ++iter;
    }
  }
  GE_IF_BOOL_EXEC(!find_node, return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

std::shared_ptr<ComputeGraph> ComputeGraphImpl::AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  if (sub_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "The graph ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The graph ptr should not be null.");
    return nullptr;
  }
  sub_graph_.push_back(sub_graph);
  names_to_subgraph_[sub_graph->GetName()] = sub_graph;
  return sub_graph;
}

graphStatus ComputeGraphImpl::RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  if (sub_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "The graph ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The graph ptr should not be null.");
    return GRAPH_FAILED;
  }

  (void)names_to_subgraph_.erase(sub_graph->GetName());
  const auto iter = find(sub_graph_.begin(), sub_graph_.end(), sub_graph);
  if (iter != sub_graph_.end()) {
    (void)sub_graph_.erase(iter);
  } else {
    GELOGW("[Remove][Subgraph] find sub_graph failed");
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddSubgraph(const std::string &name,
                                          const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Try to add a null subgraph, name %s", name.c_str());
    GE_LOGE("[Check][Param] Try to add a null subgraph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto parent_graph = subgraph->GetParentGraph();
  if (parent_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Try to add subgraph without parent graph, name %s", name.c_str());
    GE_LOGE("[Get][Graph] Try to add subgraph without parent graph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto parent_node = subgraph->GetParentNode();
  if (parent_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Try to add a subgraph without parent node, name %s", name.c_str());
    GE_LOGE("[Get][Node] Try to add a subgraph without parent node, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (parent_node->GetOwnerComputeGraph() != parent_graph) {
    REPORT_INNER_ERROR("E19999", "Try to add a subgraph which parent node's graph is not equal to "
                       "the subgraph's parent graph, subgraph name %s, parent node name %s",
                       subgraph->GetName().c_str(), parent_graph->GetName().c_str());
    GE_LOGE("[Check][Param] Try to add a subgraph which parent node's graph is not equal to "
            "the subgraph's parent graph, subgraph name %s, parent node name %s",
            subgraph->GetName().c_str(), parent_graph->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (!this->parent_graph_.expired()) {
    GELOGW("[Add][Subgraph] The subgraphs should only be added to the root graph");
  }
  if (name != subgraph->GetName()) {
    GELOGW("[Add][Subgraph] The subgraph name %s is different with input %s", subgraph->GetName().c_str(),
           name.c_str());
  }
  if (names_to_subgraph_.find(name) != names_to_subgraph_.end()) {
    REPORT_INNER_ERROR("E19999", "The subgraph %s existed", name.c_str());
    GE_LOGE("[Check][Param] The subgraph %s existed", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  sub_graph_.push_back(subgraph);
  names_to_subgraph_[name] = subgraph;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraphImpl::RemoveSubgraph(const std::string &name) {
  const auto iter = names_to_subgraph_.find(name);
  if (iter == names_to_subgraph_.end()) {
    return;
  }
  for (auto vec_iter = sub_graph_.begin(); vec_iter != sub_graph_.end(); ++vec_iter) {
    if (*vec_iter == iter->second) {
      (void)sub_graph_.erase(vec_iter);
      break;
    }
  }
  (void)names_to_subgraph_.erase(iter);
}

std::shared_ptr<ComputeGraph> ComputeGraphImpl::GetSubgraph(const std::string &name) const {
  const std::shared_ptr<ComputeGraph> parent = parent_graph_.lock();
  if (parent == nullptr) {
    const auto iter = names_to_subgraph_.find(name);
    return iter == names_to_subgraph_.end() ? nullptr : iter->second;
  } else {
    return parent->GetSubgraph(name);
  }
}

std::vector<std::shared_ptr<ComputeGraph>> ComputeGraphImpl::GetAllSubgraphs() const {
  return sub_graph_;
}

void ComputeGraphImpl::SetAllSubgraphs(const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) {
  sub_graph_ = subgraphs;
}

shared_ptr<ComputeGraph> ComputeGraphImpl::GetParentGraph() {
  return parent_graph_.lock();
}

void ComputeGraphImpl::SetParentGraph(const shared_ptr<ComputeGraph> &parent) {
  parent_graph_ = parent;
}

shared_ptr<Node> ComputeGraphImpl::GetParentNode() {
  return parent_node_.lock();
}

void ComputeGraphImpl::SetParentNode(const shared_ptr<Node> &parent) {
  parent_node_ = parent;
}

/// @brief Update input-mapping
/// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
/// @return graphStatus
graphStatus ComputeGraphImpl::UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &input : nodes_) {
    if (input->GetType() == DATA) {
      uint32_t cur_index = 0U;
      if (!ge::AttrUtils::GetInt(input->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
        continue;
      }
      const auto iter = input_mapping.find(cur_index);
      if (iter == input_mapping.end()) {
        continue;
      }
      if (!ge::AttrUtils::SetInt(input->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(iter->second))) {
        REPORT_CALL_ERROR("E19999", "set attr ATTR_NAME_PARENT_NODE_INDEX failed, op:%s.",
                          input->GetOpDesc()->GetName().c_str());
        GE_LOGE("[Call][SetInt] UpdateInputMapping failed: set attr ATTR_NAME_PARENT_NODE_INDEX failed, op:%s.",
                input->GetOpDesc()->GetName().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

/// @brief Update output-mapping
/// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
/// @return graphStatus
graphStatus ComputeGraphImpl::UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  const NodePtr net_output = FindFirstNodeMatchType(NETOUTPUT);
  if (net_output == nullptr) {
    REPORT_INNER_ERROR("E19999", "UpdateOutputMapping failed: node type %s not exist in graph.", NETOUTPUT);
    GE_LOGE("[Get][NodeType] UpdateOutputMapping failed: node type %s not exist in graph.", NETOUTPUT);
    return GRAPH_FAILED;
  }
  OpDescPtr op_desc = net_output->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "net output's op desc pr should not be null.");
    GE_LOGE("[Get][OpDesc] UpdateOutputMapping failed: op_desc is NULL.");
    return GRAPH_FAILED;
  }

  const size_t num = op_desc->GetAllInputsSize();
  for (size_t i = 0UL; i < num; i++) {
    GeTensorDesc tensor = op_desc->GetInputDesc(static_cast<uint32_t>(i));
    uint32_t cur_index = 0U;
    if (!ge::AttrUtils::GetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
      continue;
    }
    const auto iter = output_mapping.find(cur_index);
    if (iter == output_mapping.end()) {
      continue;
    }
    if (!ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(iter->second))) {
      REPORT_CALL_ERROR("E19999", "op %s set %zu input tensor attr ATTR_NAME_PARENT_NODE_INDEX failed.",
                        op_desc->GetName().c_str(), i);
      GE_LOGE("[Set][Int] op %s set %zu input tensor attr ATTR_NAME_PARENT_NODE_INDEX failed.",
              op_desc->GetName().c_str(), i);
      return GRAPH_FAILED;
    }
    if (op_desc->UpdateInputDesc(static_cast<uint32_t>(i), tensor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "op %s update %zu input_tensor failed.", op_desc->GetName().c_str(), i);
      GE_LOGE("[Update][InputDesc] UpdateOutputMapping failed: update %zu input_tensor failed.", i);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::ReorderEventNodes(const ConstComputeGraphPtr &compute_graph) {
  std::list<NodePtr> &node_list = nodes_;
  for (const auto &node : GetDirectNode(compute_graph)) {
    if (node->GetType() == RECV) {
      const auto iter = find(node_list.begin(), node_list.end(), node);
      if (iter != node_list.end()) {
        (void)node_list.erase(iter);
      }

      const auto dst_iter = find(node_list.begin(), node_list.end(), node->GetOutControlNodes().at(0UL));
      (void)node_list.insert(dst_iter, node);
    }
    if (node->GetType() == SEND) {
      const auto iter = find(node_list.begin(), node_list.end(), node);
      if (iter != node_list.end()) {
        (void)node_list.erase(iter);
      }

      auto src_iter = find(node_list.begin(), node_list.end(), node->GetInControlNodes().at(0UL));
      (void)node_list.insert(++src_iter, node);
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::InsertGraphEvents(const ConstComputeGraphPtr &compute_graph) {
  auto status = ReorderEventNodes(compute_graph);
  if (status != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Graph [%s] record event nodes failed, status:%d", name_.c_str(), status);
    GELOGE(status, "[Reorder][EventNodes] failed for Graph:%s, status:%d", name_.c_str(), status);
    return status;
  }

  // Partition subgraph
  for (const auto graph : sub_graph_) {
    status = graph->ReorderEventNodes();
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "ReorderEventNodes failed for SubGraph:%s, status:%d",
                        graph->GetName().c_str(), status);
      GELOGE(status, "[Reorder][EventNodes] failed for SubGraph:%s, status:%d", graph->GetName().c_str(), status);
      return status;
    }
  }

  std::vector<ComputeGraphPtr> subgraphs;
  const auto nodes = AllGraphNodes(subgraphs, compute_graph);
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    NodePtr node = nodes.at(i);   // [node: should not be null]
    node->GetOpDesc()->SetId(static_cast<int64_t>(i));  // [node->GetOpDesc(): should not be null]
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::DFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                    std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                    std::vector<NodePtr> &stack, const bool reverse,
                                                    const ConstComputeGraphPtr &compute_graph) {
  GELOGD("Runing_Dfs_Sort: %s", name_.c_str());
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack, map_in_edge_num, compute_graph) == GRAPH_SUCCESS,
                   return GRAPH_FAILED, "sort nodes failed");
  std::vector<NodePtr> out_nodes;
  auto stack_push = [&reverse, &stack](std::vector<NodePtr>& tmp_out_nodes) {
      if (reverse) {
        std::reverse(tmp_out_nodes.begin(), tmp_out_nodes.end());
      }
      stack.insert(stack.end(), tmp_out_nodes.begin(), tmp_out_nodes.end());
      tmp_out_nodes.clear();
  };
  // Only data nodes here
  while (!stack.empty()) {
    const NodePtr node = stack.back();
    stack.pop_back();
    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("node_vec.push_back %s", node->GetOpDesc()->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(anchor);
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
        if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
          out_nodes.push_back(peer_in_anchor->GetOwnerNode());
        }
      }
      stack_push(out_nodes);
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
        if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
          out_nodes.push_back(peer_in_anchor->GetOwnerNode());
        }
      }
      stack_push(out_nodes);
    }
    GE_IF_BOOL_EXEC(
        node->GetOutControlAnchor() != nullptr, for (const AnchorPtr peer_in_anchor
                                                     : node->GetOutControlAnchor()->GetPeerAnchors()) {
          GE_CHECK_NOTNULL(peer_in_anchor);
          const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
          if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
            out_nodes.push_back(peer_in_anchor->GetOwnerNode());
          }
        }
        stack_push(out_nodes);)
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::BFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                    std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                    std::deque<NodePtr> &stack,
                                                    const ConstComputeGraphPtr &compute_graph) {
  GELOGI("Runing_Bfs_Sort: %s", name_.c_str());
  std::vector<NodePtr> stack_input;
  std::map<std::string, NodePtr> breadth_node_map;
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack_input, map_in_edge_num, compute_graph) == GRAPH_SUCCESS,
                   return GRAPH_FAILED, "sort nodes failed");

  // Only data nodes here
  while ((!stack_input.empty()) || (!stack.empty())) {
    NodePtr node = nullptr;
    if (!stack.empty()) {
      node = stack.back();
      stack.pop_back();
    } else {
      node = stack_input.back();
      stack_input.pop_back();
    }

    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("node_vec.push_back %s", node->GetOpDesc()->GetName().c_str());
    (void)CollectBreadthOutNode(node, map_in_edge_num, breadth_node_map);

    for (const auto &name_node : breadth_node_map) {
      (void)stack.push_front(name_node.second);
    }
    breadth_node_map.clear();
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                    std::map<std::string, NodePtr> &breadth_node_map) {
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }

    for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    for (const AnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerAnchors()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end() && (--iter->second == 0)) {
        (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetOwnerNode());
      }
    }
  }
  return GRAPH_SUCCESS;
}

void ComputeGraphImpl::TopologicalSorting(std::function<bool (const NodePtr &, const NodePtr &)> comp) {
  nodes_.sort(std::move(comp));
  int64_t num = 0;
  for (const NodePtr &node : nodes_) {
    node->GetOpDesc()->SetId(num++);  // node should not be null, node->GetOpDesc() should not be null]
  }
}

graphStatus ComputeGraphImpl::TopologicalSorting(const ComputeGraphPtr &const_graph_ptr,
                                                 const ConstComputeGraphPtr &const_compute_graph) {
  auto ret = TopologicalSortingGraph(const_compute_graph);
  if (ret != GRAPH_SUCCESS) {
    GE_DUMP(const_graph_ptr, "black_box" + name_);
    REPORT_CALL_ERROR("E19999", "Graph [%s] topological sort failed, saved to file black_box", name_.c_str());
    GELOGE(ret, "[Sort][Graph] Graph [%s] topological sort failed, saved to file black_box", name_.c_str());
    return ret;
  }

  if (sub_graph_.empty()) {
    return GRAPH_SUCCESS;
  }

  // partition sub graph
  for (const auto &sub_graph : sub_graph_) {
    ret = sub_graph->TopologicalSortingGraph();
    if (ret != GRAPH_SUCCESS) {
      GE_DUMP(sub_graph, "black_box" + sub_graph->GetName());
      REPORT_CALL_ERROR("E19999", "Sub graph[%s] topological sort failed, saved to file black_box",
                        sub_graph->GetName().c_str());
      GELOGE(ret, "[Sort][Graph] Sub graph[%s] topological sort failed, saved to file black_box",
             sub_graph->GetName().c_str());
      return ret;
    }
  }

  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  auto nodes = AllGraphNodes(subgraphs, const_compute_graph);
  for (size_t i = 0UL; i < nodes.size(); i++) {
    NodePtr node = nodes.at(i);   // [node: should not be null]
    node->GetOpDesc()->SetId(static_cast<int64_t>(i));  // [node->GetOpDesc(): should not be null]
  }
  if (sub_graph_.size() != subgraphs.size()) {  // Graph Partition use subgraph, Keep original
    GELOGW("[TopoSort][CheckNodeSize] Keep original subgraph for graph size %zu not equal %zu.", sub_graph_.size(),
           subgraphs.size());
    return GRAPH_SUCCESS;
  }
  sub_graph_.swap(subgraphs);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph,
                                                      const bool dfs_reverse) {
  std::vector<NodePtr> node_vec;
  std::map<NodePtr, uint32_t> map_in_edge_num;
  const bool use_BFS = IsUseBFS();
  if (use_BFS) {
    std::deque<NodePtr> stack;
    if (BFSTopologicalSorting(node_vec, map_in_edge_num, stack, compute_graph) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  } else {
    std::vector<NodePtr> stack;
    if (DFSTopologicalSorting(node_vec, map_in_edge_num, stack, dfs_reverse, compute_graph) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  // If they are not equal, there is a closed loop
  if (node_vec.size() != GetDirectNodesSize()) {
    std::set<Node *> itered_nodes_set;
    for (auto &node : node_vec) {
      (void)itered_nodes_set.insert(node.get());
    }
    REPORT_INNER_ERROR("E19999", "Failed to do topo sorting total %zu, itered %zu, exist closed loop in graph:%s",
                       GetDirectNodesSize(), node_vec.size(), name_.c_str());
    GE_LOGE("[Check][Param] Failed to do topo sorting total %zu, itered %zu, exist closed loop in graph.",
            GetDirectNodesSize(), node_vec.size());
    for (auto &node : nodes_) {
      if (itered_nodes_set.count(node.get()) == 0UL) {
        GE_LOGE("[Check][Param] The node %s does not itered when topological sorting", node->GetName().c_str());
      }
    }
    return GRAPH_FAILED;
  }

  ClearNodeList();
  for (size_t i = 0UL; i < node_vec.size(); i++) {
    NodePtr node = node_vec[i];   // [node: should not be null]
    node->GetOpDesc()->SetId(static_cast<int64_t>(i));  // [node->GetOpDesc(): should not be null]
    PushBackToNodeList(node);
  }

  is_valid_flag_ = true;
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::SortNodes(std::vector<NodePtr> &stack,
                                        std::map<NodePtr, uint32_t> &map_in_edge_num,
                                        const ConstComputeGraphPtr &compute_graph) {
  // Record the number of non data nodes but no input nodes
  uint32_t spec_node_size = 0U;
  for (const auto &node : GetDirectNode(compute_graph)) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    map_in_edge_num[node] = static_cast<uint32_t>(GetInEdgeSize(node));
    if (map_in_edge_num[node] == 0U) {
      if ((node->GetOpDesc()->GetType() != DATA) && (node->GetOpDesc()->GetType() != AIPPDATA) &&
          (node->GetOpDesc()->GetType() != INPUT_TYPE) && (node->GetOpDesc()->GetType() != ANN_DATA)) {
        (void)stack.insert(stack.begin(), node);
        spec_node_size++;
        continue;
      }
      // Need to insert the data nodes in reverse order
      (void)stack.insert(stack.begin() + static_cast<int64_t>(spec_node_size), node);
    }
  }

  /// Make sure the inputs order matches with user-designated
  /// 1. Get the index of two input nodes in the user-inputs-order(inputs_order_)
  /// 2. Compare two indices, if not match, swap the positions of two inputs
  /// *: Remind: stack is reverse-order
  for (size_t i = 0UL; i < stack.size(); ++i) {
    // If not found in 'inputs_order_', skip it
    const auto it_i = std::find(inputs_order_.begin(), inputs_order_.end(), stack[i]->GetName());
    GE_IF_BOOL_EXEC(it_i == inputs_order_.end(), continue);
    const auto inx_i = it_i - inputs_order_.begin();
    for (size_t j = i + 1UL; j < stack.size(); ++j) {
      // If not found in 'inputs_order_', skip it
      const auto it_j = std::find(inputs_order_.begin(), inputs_order_.end(), stack[j]->GetName());
      GE_IF_BOOL_EXEC(it_j == inputs_order_.end(), continue);

      // Compare index, swap them if it should be
      const auto inx_j = it_j - inputs_order_.begin();
      GE_IF_BOOL_EXEC(inx_i < inx_j, std::swap(stack[i], stack[j]));
    }
  }

  return GRAPH_SUCCESS;
}

size_t ComputeGraphImpl::GetInEdgeSize(const NodePtr &node) {
  size_t in_edge_size = 0UL;
  if (node == nullptr) {
    return in_edge_size;
  }
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    in_edge_size = in_edge_size + anchor->GetPeerAnchorsSize();
    // Break flow control data loop.
    const OutDataAnchorPtr out_anchor = anchor->GetPeerOutAnchor();
    if ((out_anchor != nullptr) && (out_anchor->GetOwnerNode() != nullptr)) {
      const NodePtr out_node = out_anchor->GetOwnerNode();
      if ((out_node->GetType() == NEXTITERATION) || (out_node->GetType() == REFNEXTITERATION)) {
        GE_IF_BOOL_EXEC(in_edge_size == 0UL,
                        GELOGE(GRAPH_FAILED, "[Check][Param] If [in_edge_size = 0], the result will be reversed");
                        return in_edge_size);
        in_edge_size -= 1UL;
      }
    }
  }
  if (node->GetInControlAnchor() != nullptr) {
    in_edge_size = in_edge_size + node->GetInControlAnchor()->GetPeerAnchorsSize();
  }
  return in_edge_size;
}

size_t ComputeGraphImpl::GetOutEdgeSize(const NodePtr &node) {
  size_t out_edge_size = 0UL;
  if (node == nullptr) {
    return out_edge_size;
  }

  // Break flow control data loop.
  if ((node->GetType() != NEXTITERATION) && (node->GetType() != REFNEXTITERATION)) {
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      if (anchor != nullptr) {
        out_edge_size = out_edge_size + anchor->GetPeerAnchors().size();
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    if (out_edge_size > (UINT64_MAX - node->GetOutControlAnchor()->GetPeerAnchors().size())) {
      return 0UL;
    }
    out_edge_size = out_edge_size + node->GetOutControlAnchor()->GetPeerAnchors().size();
  }
  return out_edge_size;
}

bool ComputeGraphImpl::IsValid() const { return is_valid_flag_; }

void ComputeGraphImpl::InValid() { is_valid_flag_ = false; }

void ComputeGraphImpl::Dump(const ConstComputeGraphPtr &compute_graph) const {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return;
  }

  GELOGI("graph name = %s.", GetName().c_str());
  for (const auto &node : GetAllNodes(compute_graph)) {
    GELOGD("node name = %s.", node->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out data node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
    const auto out_control_anchor = node->GetOutControlAnchor();
    if (out_control_anchor != nullptr) {
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC(peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNode() != nullptr,
                        GELOGI("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
  }
}

void ComputeGraphImpl::Swap(ComputeGraphImpl &graph) {
  origGraph_.swap(graph.origGraph_);

  name_.swap(graph.name_);
  std::swap(graph_id_, graph.graph_id_);
  attrs_.Swap(graph.attrs_);
  nodes_.swap(graph.nodes_);
  const auto tmp_size = direct_nodes_size_;
  direct_nodes_size_ = graph.direct_nodes_size_;
  graph.direct_nodes_size_ = tmp_size;
  all_nodes_infos_.swap(graph.all_nodes_infos_);
  target_nodes_info_.swap(graph.target_nodes_info_);

  input_nodes_.swap(graph.input_nodes_);
  inputs_order_.swap(graph.inputs_order_);
  std::swap(input_size_, graph.input_size_);
  out_nodes_map_.swap(graph.out_nodes_map_);
  std::swap(output_size_, graph.output_size_);
  output_nodes_info_.swap(graph.output_nodes_info_);

  sub_graph_.swap(graph.sub_graph_);
  names_to_subgraph_.swap(graph.names_to_subgraph_);
  parent_graph_.swap(graph.parent_graph_);
  parent_node_.swap(graph.parent_node_);

  // the members followed should not in the ComputeGraphImpl class
  std::swap(is_valid_flag_, graph.is_valid_flag_);
  std::swap(is_summary_graph_, graph.is_summary_graph_);
  std::swap(need_iteration_, graph.need_iteration_);
  params_share_map_.swap(graph.params_share_map_);
  op_name_map_.swap(graph.op_name_map_);
  std::swap(session_id_, graph.session_id_);
  std::swap(data_format_, graph.data_format_);
  std::swap(is_unknown_shape_graph_, graph.is_unknown_shape_graph_);
}

void ComputeGraphImpl::SetNodesOwner(const ComputeGraphPtr &compute_graph) {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    (void)node->SetOwnerComputeGraph(compute_graph);
  }
}

graphStatus ComputeGraphImpl::IsolateNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  const auto next_nodes = node->GetOutAllNodes();
  // If there is input data side
  for (size_t i = 0UL; i < node->GetAllInDataAnchors().size(); i++) {
    const auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    const auto pre_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (pre_out_data_anchor != nullptr) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_data_anchor, in_data_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                         pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                         in_data_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                       pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                       in_data_anchor->GetOwnerNode()->GetName().c_str());
      GE_IF_BOOL_EXEC(pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANT ||
                      pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANTOP,
                      continue);
      for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
        for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                             out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                             next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                           out_data_anchor->GetOwnerNode()->GetName().c_str(),
                           next_in_data_anchor->GetOwnerNode()->GetName().c_str());
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                             pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                             next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                           pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                           next_in_data_anchor->GetOwnerNode()->GetName().c_str());
        }
        for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                             out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                             next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                           out_data_anchor->GetOwnerNode()->GetName().c_str(),
                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                             pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                             next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                           pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        }
      }
      const auto out_ctrl_anchor = node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(out_ctrl_anchor);
      const auto pre_out_ctrl_anchor = pre_out_data_anchor->GetOwnerNode()->GetOutControlAnchor();
      GE_CHECK_NOTNULL(pre_out_ctrl_anchor);
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                           out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                         out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                           pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                         pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }

  // If there is an input control side
  const auto in_ctrl_anchor = node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  for (const auto &pre_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_ctrl_anchor, in_ctrl_anchor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                       pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                       in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                     return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                     pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                     in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                           out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                         out_data_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                           pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                         pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
    const auto out_ctrl_anchor = node->GetOutControlAnchor();
    if (out_ctrl_anchor != nullptr) {
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                           out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                         out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                           pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                           next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                         pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }

  for (const auto &out_peer_data_anchor : in_ctrl_anchor->GetPeerOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_peer_data_anchor, in_ctrl_anchor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                       out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                                       in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                     return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                     out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                     in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    for (const auto &next_node : next_nodes) {
      const auto next_in_control_anchor = next_node->GetInControlAnchor();
      GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(out_peer_data_anchor, next_in_control_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed",
                                         out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                                         next_in_control_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                       out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                       next_in_control_anchor->GetOwnerNode()->GetName().c_str());
    }
  }

  return RemoveExtraOutEdge(node);
}

graphStatus ComputeGraphImpl::RemoveExtraOutEdge(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  // Remove redundant output edges
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                         out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                         next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                       out_data_anchor->GetOwnerNode()->GetName().c_str(),
                       next_in_data_anchor->GetOwnerNode()->GetName().c_str());
    }

    for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                         out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                       out_data_anchor->GetOwnerNode()->GetName().c_str(),
                       next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    }
  }
  const auto out_ctrl_anchor = node->GetOutControlAnchor();
  if (out_ctrl_anchor != nullptr) {
    for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed",
                                         out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                         next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                       out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                       next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::Verify(const ConstComputeGraphPtr compute_graph) {
  const bool is_unknown_graph = GetGraphUnknownFlag();
  for (const auto &node_ptr : GetAllNodes(compute_graph)) {
    GE_CHECK_NOTNULL(node_ptr);
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    GE_IF_BOOL_EXEC(is_unknown_graph, continue);
    GE_CHK_BOOL_EXEC(node_ptr->GetOpDesc()->CommonVerify() == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Verifying %s failed.", node_ptr->GetName().c_str());
                     return GRAPH_FAILED, "[Call][CommonVerify] Verifying %s failed.", node_ptr->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::InferShapeInNeed(const ComputeGraphPtr &const_graph_ptr,
                                               const ConstComputeGraphPtr &const_compute_graph) {
  GE_CHK_BOOL_ONLY_LOG(TopologicalSorting(const_graph_ptr, const_compute_graph) == GRAPH_SUCCESS, "Verifying failed.");
  for (const auto &node_ptr : GetAllNodes(const_compute_graph)) {
    GE_CHECK_NOTNULL(node_ptr);
    auto op_desc = node_ptr->GetOpDesc();
    bool is_need_infer = false;
    (void)ge::AttrUtils::GetBool(op_desc, NEED_INFER, is_need_infer);
    if (is_need_infer) {
      GE_CHK_BOOL_EXEC(node_ptr->Verify() == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "Verifying %s failed.", node_ptr->GetName().c_str());
                       return GRAPH_FAILED, "[Call][Verify] Verifying %s failed.", node_ptr->GetName().c_str());

      const graphStatus status = node_ptr->InferShapeAndType();
      GE_CHK_BOOL_EXEC_INFO((node_ptr->GetType() == DATA) || (GRAPH_PARAM_INVALID != status), break,
                            "Op %s does not have the IMPLEMT_INFERFUNC definition,"
                            " and subsequent operators no longer perform shape inference.",
                            node_ptr->GetName().c_str());
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "Inferring %s failed.", node_ptr->GetName().c_str());
                       return GRAPH_FAILED, "[Call][InferShapeAndType] Inferring %s failed.",
                       node_ptr->GetName().c_str());

      for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
        auto output_tensor = out_anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(
            static_cast<uint32_t>(out_anchor->GetIdx()));
        ge::TensorUtils::SetRealDimCnt(output_tensor, static_cast<uint32_t>(output_tensor.GetShape().GetDims().size()));
        (void)out_anchor->GetOwnerNode()->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(out_anchor->GetIdx()),
                                                                        output_tensor);
        for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
          (void)peer_anchor->GetOwnerNode()->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(peer_anchor->GetIdx()),
                                                                          output_tensor);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

ProtoAttrMap &ComputeGraphImpl::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &ComputeGraphImpl::GetAttrMap() const {
  return attrs_;
}

const std::map<OperatorImplPtr, NodePtr> &ComputeGraphImpl::GetAllNodesInfo() const { return all_nodes_infos_; }

void ComputeGraphImpl::SetUserDefOutput(const std::string &output_name) {
  if (output_name.empty()) {
    return;
  }

  const std::vector<std::string> nodes = StringUtils::Split(output_name, ';');
  for (const std::string node : nodes) {
    std::vector<std::string> item = StringUtils::Split(node, ':');
    if (item.size() != OUTPUT_PARAM_SIZE) {
      REPORT_INNER_ERROR("W19999", "Check output param size failed, output_name:%s", output_name.c_str());
      GELOGW("[Check][Output] Check output param size failed, output_name:%s", output_name.c_str());
      continue;
    }

    int32_t index;
    try {
      index = stoi(StringUtils::Trim(item[1UL]));
    } catch (const std::out_of_range &) {
      REPORT_INNER_ERROR("W19999", "Catch out_of_range exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch out_of_range exception, output_name:%s", output_name.c_str());
      continue;
    } catch (const std::invalid_argument &) {
      REPORT_INNER_ERROR("W19999", "Catch invalid_argument exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch invalid_argument exception, output_name:%s", output_name.c_str());
      continue;
    } catch (...) {
      REPORT_INNER_ERROR("W19999", "Catch exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch exception, output_name:%s", output_name.c_str());
      continue;
    }
    const auto iter = out_nodes_map_.find(item[0UL]);
    if (iter == out_nodes_map_.end()) {
      out_nodes_map_[item[0UL]] = std::vector<int32_t>(1UL, index);
    } else {
      const auto idx_iter = std::find(iter->second.begin(), iter->second.end(), index);
      if (idx_iter == iter->second.end()) {
        iter->second.push_back(index);
      }
    }
  }
}

const std::string ComputeGraphImpl::GetOutput() {
  static const int32_t resultDefaultSize = 2048;
  std::string result;
  result.reserve(static_cast<uint64_t>(resultDefaultSize));
  auto iter = out_nodes_map_.begin();
  while (iter != out_nodes_map_.end()) {
    const auto idxes = iter->second;
    for (const auto idx : idxes) {
      (void)result.append(iter->first).append(":").append(std::to_string(idx)).append(";");
    }
    ++iter;
  }

  return result.substr(0UL, result.length() - 1UL);
}


void ComputeGraphImpl::EraseFromNodeList(const std::list<NodePtr>::iterator &position) {
  (void) nodes_.erase(position);
  --direct_nodes_size_;
}

void ComputeGraphImpl::InsertToNodeList(const std::list<NodePtr>::iterator &position, const NodePtr &node) {
  (void) nodes_.insert(position, node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::PushBackToNodeList(const NodePtr &node) {
  (void) nodes_.push_back(node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::EmplaceBackToNodeList(const NodePtr &node) {
  (void) nodes_.emplace_back(node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::ClearNodeList() {
  (void) nodes_.clear();
  direct_nodes_size_ = 0UL;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const std::string &name)
    : enable_shared_from_this(),
      AttrHolder(),
      impl_(ComGraphMakeShared<ComputeGraphImpl>(name)) {}

ComputeGraph::~ComputeGraph() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const ge::ComputeGraph& compute_graph)
    : enable_shared_from_this(),
      AttrHolder(compute_graph),
      impl_(ComGraphMakeShared<ComputeGraphImpl>(*(compute_graph.impl_))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(ge::ComputeGraph&& compute_graph)
    : enable_shared_from_this(),
      AttrHolder(std::move(compute_graph)),
      impl_(ComGraphMakeShared<ComputeGraphImpl>(std::move(*(compute_graph.impl_)))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string ComputeGraph::GetName() const { return impl_->GetName(); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetName(const std::string &name) {
  impl_->SetName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t ComputeGraph::GetAllNodesSize() const {
  return GetAllNodes().size();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetAllNodes() const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodes(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr>
ComputeGraph::GetAllNodes(const NodeFilter &node_filter, const GraphFilter &graph_filter) const {
  return impl_->GetAllNodes(node_filter, graph_filter, shared_from_this());
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs) const {
  return impl_->AllGraphNodes(subgraphs, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ComputeGraph::Vistor<NodePtr> ComputeGraph::GetNodes(const bool is_unknown_shape) const {
  return impl_->GetNodes(is_unknown_shape, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr>
ComputeGraph::GetNodes(const bool is_unknown_shape, const NodeFilter &node_filter,
                       const GraphFilter &graph_filter) const {
  return impl_->GetNodes(is_unknown_shape, node_filter, graph_filter, shared_from_this());
}

size_t ComputeGraph::GetDirectNodesSize() const {
  return impl_->GetDirectNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetDirectNode() const {
  return impl_->GetDirectNode(shared_from_this());
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetInputNodes() const {
  return impl_->GetInputNodes(shared_from_this());
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetOutputNodes() const {
  return impl_->GetOutputNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::FindNode(const std::string &name) const {
  return impl_->FindNode(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
NodePtr ComputeGraph::FindFirstNodeMatchType(const std::string &name) const {
  return impl_->FindFirstNodeMatchType(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphAttrsAreEqual(
    const ComputeGraph &r_graph) const {
  return impl_->GraphAttrsAreEqual(*(r_graph.impl_));
}

/// Since there may be different input nodes
/// chosen by user in the same graph, special judgment is needed
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::VectorInputNodePtrIsEqual(
    const std::vector<NodePtr> &left_nodes, const std::vector<NodePtr> &right_nodes) const {
  return impl_->VectorInputNodePtrIsEqual(left_nodes, right_nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphMembersAreEqual(
    const ComputeGraph &r_graph) const {
  return impl_->GraphMembersAreEqual(*(r_graph.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::operator==(const ComputeGraph &r_graph) const {
  return *impl_ == *(r_graph.impl_);
}

ComputeGraph& ComputeGraph::operator=(ge::ComputeGraph compute_graph) {
  if (&compute_graph == this) {
    return *this;
  }
  AttrHolder::Swap(compute_graph);
  *impl_ = *(compute_graph.impl_);
  return *this;
}

NodePtr ComputeGraph::AddNodeFront(const NodePtr node) {
  return impl_->AddNodeFront(node);
}

NodePtr ComputeGraph::AddNodeFront(const OpDescPtr &op) {
  return impl_->AddNodeFront(op, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(NodePtr node) {
  return impl_->AddNode(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(OpDescPtr op) {
  return impl_->AddNode(op, shared_from_this());
}

NodePtr ComputeGraph::AddNode(OpDescPtr op, const int64_t id) {  // for unserialize.
  return impl_->AddNode(op, id, shared_from_this());
}

NodePtr ComputeGraph::AddInputNode(const NodePtr node) {
  return impl_->AddInputNode(node);
}

NodePtr ComputeGraph::AddOutputNode(const NodePtr node) {
  return AddOutputNodeByIndex(node, 0);
}

NodePtr ComputeGraph::AddOutputNodeByIndex(const NodePtr node, const int32_t index) {
  return impl_->AddOutputNodeByIndex(node, index);
}

graphStatus ComputeGraph::RemoveConstInput(const NodePtr &node) {
  return impl_->RemoveConstInput(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::RemoveNode(const NodePtr &node) {
  return impl_->RemoveNode(node);
}

// Used in sub_graph scenes
graphStatus ComputeGraph::RemoveInputNode(const NodePtr &node) {
  return impl_->RemoveInputNode(node);
}

graphStatus ComputeGraph::RemoveOutputNode(const NodePtr &node) {
  return impl_->RemoveOutputNode(node);
}

std::shared_ptr<ComputeGraph> ComputeGraph::AddSubGraph(const std::shared_ptr<ComputeGraph> sub_graph) {
  return impl_->AddSubGraph(sub_graph);
}

graphStatus ComputeGraph::RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  return impl_->RemoveSubGraph(sub_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph) {
  return impl_->AddSubgraph(name, subgraph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  return AddSubgraph(subgraph->GetName(), subgraph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(const std::string &name) {
  return impl_->RemoveSubgraph(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(
    const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph != nullptr) {
    RemoveSubgraph(subgraph->GetName());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<ComputeGraph> ComputeGraph::GetSubgraph(
    const std::string &name) const {
  return impl_->GetSubgraph(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::shared_ptr<ComputeGraph>>
ComputeGraph::GetAllSubgraphs() const {
  return impl_->GetAllSubgraphs();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetAllSubgraphs(
    const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) {
  return impl_->SetAllSubgraphs(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::map<std::vector<std::string>, std::vector<std::string>> &ComputeGraph::GetShareParamLayer() const {
  return impl_->GetShareParamLayer();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetShareParamLayer(
    const std::map<std::vector<std::string>, std::vector<std::string>> params_share_map) {
  impl_->SetShareParamLayer(params_share_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetInputsOrder(
    const std::vector<std::string> &inputs_order) {
  impl_->SetInputsOrder(inputs_order);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphOutNodes(
    const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
  impl_->SetGraphOutNodes(out_nodes_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::AppendGraphOutNodes(
    const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
  impl_->AppendGraphOutNodes(out_nodes_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY shared_ptr<ComputeGraph> ComputeGraph::GetParentGraph() {
  return impl_->GetParentGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentGraph(
    const shared_ptr<ComputeGraph> &parent) {
  impl_->SetParentGraph(parent);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY shared_ptr<Node> ComputeGraph::GetParentNode() {
  return impl_->GetParentNode();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentNode(const shared_ptr<Node> &parent) {
  return impl_->SetParentNode(parent);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::map<std::string, std::vector<int32_t>> &ComputeGraph::GetGraphOutNodes() const {
  return impl_->GetGraphOutNodes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetOrigGraph(const ComputeGraphPtr orig_graph) {
  impl_->SetOrigGraph(orig_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr ComputeGraph::GetOrigGraph(void) {
  return impl_->GetOrigGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetOutputSize(const uint32_t size) {
  impl_->SetOutputSize(size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetOutputSize() const {
  return impl_->GetOutputSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetInputSize(const uint32_t size) {
  impl_->SetInputSize(size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetInputSize() const {
  return impl_->GetInputSize();
}

// false: known shape  true: unknow shape
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GetGraphUnknownFlag() const {
  return impl_->GetGraphUnknownFlag();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphUnknownFlag(const bool flag) {
  impl_->SetGraphUnknownFlag(flag);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetNeedIteration(const bool need_iteration) {
  impl_->SetNeedIteration(need_iteration);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GetNeedIteration() const {
  return impl_->GetNeedIteration();
}

///
/// @brief Update input-mapping
/// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  return impl_->UpdateInputMapping(input_mapping);
}

///
/// @brief Update output-mapping
/// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  return impl_->UpdateOutputMapping(output_mapping);
}

graphStatus ComputeGraph::ReorderEventNodes() {
  return impl_->ReorderEventNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InsertGraphEvents() {
  return impl_->InsertGraphEvents(shared_from_this());
}

graphStatus ComputeGraph::DFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::vector<NodePtr> &stack, const bool reverse) {
  return impl_->DFSTopologicalSorting(node_vec, map_in_edge_num, stack, reverse, shared_from_this());
}

graphStatus ComputeGraph::BFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::deque<NodePtr> &stack) {
  return impl_->BFSTopologicalSorting(node_vec, map_in_edge_num, stack, shared_from_this());
}

graphStatus ComputeGraph::CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::map<std::string, NodePtr> &breadth_node_map) {
  return impl_->CollectBreadthOutNode(node, map_in_edge_num, breadth_node_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::TopologicalSorting(
    std::function<bool (const NodePtr &, const NodePtr &)> comp) {
  return impl_->TopologicalSorting(comp);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::TopologicalSorting() {
  return impl_->TopologicalSorting(shared_from_this(), shared_from_this());
}

graphStatus ComputeGraph::TopologicalSortingGraph(const bool dfs_reverse) {
  return impl_->TopologicalSortingGraph(shared_from_this(), dfs_reverse);
}

graphStatus ComputeGraph::SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num) {
  return impl_->SortNodes(stack, map_in_edge_num, shared_from_this());
}

size_t ComputeGraph::GetInEdgeSize(const NodePtr &node) {
  return impl_->GetInEdgeSize(node);
}

size_t ComputeGraph::GetOutEdgeSize(const NodePtr &node) {
  return impl_->GetOutEdgeSize(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::IsValid() const {
  return impl_->IsValid();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  void ComputeGraph::InValid() {
  impl_->InValid();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Dump() const {
  return impl_->Dump(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Swap(ComputeGraph &graph) {
  this->AttrHolder::Swap(graph);
  impl_->Swap(*(graph.impl_));

  // Update Node owner.
  SetNodesOwner();
  graph.SetNodesOwner();
}

void ComputeGraph::SetNodesOwner() {
  return impl_->SetNodesOwner(shared_from_this());
}

void ComputeGraph::EraseFromNodeList(const std::list<NodePtr>::iterator position) {
  impl_->EraseFromNodeList(position);
}

void ComputeGraph::InsertToNodeList(const std::list<NodePtr>::iterator position, const NodePtr &node) {
  impl_->InsertToNodeList(position, node);
}

void ComputeGraph::PushBackToNodeList(const NodePtr &node) {
  impl_->PushBackToNodeList(node);
}

void ComputeGraph::EmplaceBackToNodeList(const NodePtr &node) {
  impl_->EmplaceBackToNodeList(node);
}

void ComputeGraph::ClearNodeList() {
  impl_->ClearNodeList();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::IsolateNode(const NodePtr &node) {
  return impl_->IsolateNode(node);
}

graphStatus ComputeGraph::RemoveExtraOutEdge(const NodePtr &node) {
  return impl_->RemoveExtraOutEdge(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::Verify() {
  return impl_->Verify(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InferOriginFormat() {
  return ge::FormatRefiner::InferOrigineFormat(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InferShapeInNeed() {
  return impl_->InferShapeInNeed(shared_from_this(), shared_from_this());
}

ProtoAttrMap &ComputeGraph::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMap &ComputeGraph::GetAttrMap() const {
  return impl_->GetAttrMap();
}

const std::map<OperatorImplPtr, NodePtr> &ComputeGraph::GetAllNodesInfo() const {
  return impl_->GetAllNodesInfo();
}

void ComputeGraph::SetUserDefOutput(const std::string &output_name) {
  impl_->SetUserDefOutput(output_name);
}

const std::string ComputeGraph::GetOutput() {
  return impl_->GetOutput();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphOpName(
    const std::map<uint32_t, std::string> &op_name_map) {
  impl_->SetGraphOpName(op_name_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::map<uint32_t, std::string> &ComputeGraph::GetGraphOpName() const {
  return impl_->GetGraphOpName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetAllNodesInfo(
    const std::map<OperatorImplPtr, NodePtr> &nodes) {
  impl_->SetAllNodesInfo(nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphOutNodesInfo(
    std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
  impl_->SetGraphOutNodesInfo(out_nodes_info);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::AppendGraphOutNodesInfo(
    std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
  impl_->AppendGraphOutNodesInfo(out_nodes_info);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::vector<std::pair<NodePtr, int32_t>> &ComputeGraph::GetGraphOutNodesInfo() const {
  return impl_->GetGraphOutNodesInfo();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphTargetNodesInfo(
    const std::vector<NodePtr> &target_nodes_info) {
  impl_->SetGraphTargetNodesInfo(target_nodes_info);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::vector<NodePtr> &ComputeGraph::GetGraphTargetNodesInfo() const {
  return impl_->GetGraphTargetNodesInfo();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetSessionID(const uint64_t session_id) {
  impl_->SetSessionID(session_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint64_t ComputeGraph::GetSessionID() const {
  return impl_->GetSessionID();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphID(const uint32_t graph_id) {
  impl_->SetGraphID(graph_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetGraphID() const {
  return impl_->GetGraphID();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SaveDataFormat(const ge::Format data_format) {
  impl_->SaveDataFormat(data_format);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ge::Format ComputeGraph::GetDataFormat() const {
  return impl_->GetDataFormat();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::IsSummaryGraph() const {
  return impl_->IsSummaryGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetSummaryFlag(const bool is_summary_graph) {
  impl_->SetSummaryFlag(is_summary_graph);
}
}  // namespace ge
