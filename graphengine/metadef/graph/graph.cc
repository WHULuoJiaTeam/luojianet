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

#include "external/graph/graph.h"
#include <cstring>
#include "debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/node_utils.h"


namespace {
const uint32_t kSubgraphIndexOfPartitionedCall = 0U;
}  // namespace

namespace ge {
class GraphImpl {
 public:
  friend class GraphUtils;
  GraphImpl(const GraphImpl &) = delete;
  GraphImpl &operator=(const GraphImpl &) = delete;

  explicit GraphImpl(const std::string &name) : name_(name) {}

  ~GraphImpl() {
    if (IsValid()) {
      if (compute_graph_ != nullptr) {
        GraphUtils::BreakConnect(compute_graph_->GetAllNodesInfo());
      }
    }
    for (const auto &it : op_list_) {
      const Operator op = it.second;
      op.BreakConnect();
    }
  }

  graphStatus SetInputs(const std::vector<Operator> &inputs) {
    compute_graph_ = GraphUtils::CreateGraphFromOperator(name_, inputs);
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Build][Graph] failed.");
    GE_CHK_BOOL_RET_STATUS(inputs.size() != 0U, GRAPH_FAILED, "[Check][Param] set input NULL.");
    compute_graph_->SetInputSize(static_cast<uint32_t>(inputs.size()));
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<Operator> &outputs) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERROR("E19999", "compute graph is nullptr, check invalid.");
      GELOGE(GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (outputs.empty()) {
      GELOGI("Set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    for (size_t i = 0U; i < outputs.size(); ++i) {
      output_indexs.emplace_back(outputs[i], std::vector<size_t>{});
    }

    const graphStatus ret = SetOutputs(output_indexs);
    return ret;
  }

  graphStatus SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERROR("E19999", "compute graph is nullptr, check invalid.");
      GELOGE(GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (output_indexs.empty()) {
      GELOGW("[SetOutputs][CheckParam] Set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes;
    for (const auto &item : output_indexs) {
      const Operator &output = item.first;
      const std::vector<size_t> &indexs = item.second;
      ge::NodePtr node = compute_graph_->FindNode(output.GetName());
      if (node == nullptr) {
        GELOGW("[SetOutputs][Check] User designated out_node %s not exist in graph, skip it",
               output.GetName().c_str());
        continue;
      }

      const ge::OpDescPtr tmp_op_ptr = node->GetOpDesc();
      if (tmp_op_ptr == nullptr) {
        GELOGE(GRAPH_FAILED, "op_desc in node must not be null.");
        continue;
      }
      const size_t out_size = tmp_op_ptr->GetOutputsSize();
      if (indexs.empty()) {
        for (size_t i = 0U; i < out_size; ++i) {
          output_name_ += output.GetName() + ":" + std::to_string(i) + ";";
          output_nodes.emplace_back(node, i);
        }
      } else {
        for (size_t i = 0U; i < indexs.size(); ++i) {
          if (indexs[i] >= out_size) {
            GELOGW("[SetOutputs][Check] User designated out_node %s has no output %zu, output_size=%zu, skip it",
                   output.GetName().c_str(), indexs[i], out_size);
          } else {
            output_name_ += output.GetName() + ":" + std::to_string(i) + ";";
            output_nodes.emplace_back(node, indexs[i]);
          }
        }
      }
    }

    // Del last ";"
    if (!output_name_.empty()) {
        output_name_ = output_name_.substr(0U, output_name_.length() - 1U);
    }
    compute_graph_->SetUserDefOutput(output_name_);
    compute_graph_->SetOutputSize(static_cast<uint32_t>(output_indexs.size()));
    compute_graph_->SetGraphOutNodesInfo(output_nodes);
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<std::pair<Operator, std::string>> &outputs) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Check][Param] set ComputeGraph faild.");
    GE_CHK_BOOL_EXEC_INFO(outputs.size() != 0, return GRAPH_SUCCESS, "set outputs size is 0.");

    // Construct specified output
    std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes;
    for (const auto item : outputs) {
      ge::NodePtr node = compute_graph_->FindNode(item.first.GetName());
      if (node == nullptr) {
        REPORT_INNER_ERROR("E19999", "designated out_node (%s) not exist in graph:%s, this out_node ignored!",
                           item.first.GetName().c_str(), compute_graph_->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Warning, user designated out_node (%s) not exist in graph:%s, "
               "this out_node ignored!", item.first.GetName().c_str(), compute_graph_->GetName().c_str());
        return GRAPH_FAILED;
      }
      const ge::OpDescPtr tmp_op_ptr = node->GetOpDesc();
      if (tmp_op_ptr == nullptr) {
        GELOGE(GRAPH_FAILED, "op_desc_ptr in node must not be null.");
        continue;
      }
      const size_t out_size = tmp_op_ptr->GetOutputsSize();

      if (item.second.empty()) {
        for (size_t i = 0U; i < out_size; ++i) {
          output_name_ += item.first.GetName() + ":" + std::to_string(i) + ";";
          output_nodes.push_back(std::make_pair(node, i));
        }
      } else {
        int32_t index = tmp_op_ptr->GetOutputIndexByName(item.second);
        if (index < 0) {
          REPORT_INNER_ERROR("E19999", "user designated out_node (%s):(%s) not exist in graph:%s, "
                             "this out_node ignored!", item.first.GetName().c_str(), item.second.c_str(),
                             compute_graph_->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Check][Param] Warning, user designated out_node (%s):(%s) not exist in graph:%s, "
                 "this out_node ignored!", item.first.GetName().c_str(), item.second.c_str(),
                 compute_graph_->GetName().c_str());
          return GRAPH_FAILED;
        }
        output_name_ += item.first.GetName() + ":" + std::to_string(index) + ";";
        output_nodes.push_back(std::make_pair(node, index));
      }
    }
    // Del last ";"
    if (!output_name_.empty()) {
      output_name_ = output_name_.substr(0U, output_name_.length() - 1U);
    }
    compute_graph_->SetOutputSize(static_cast<uint32_t>(outputs.size()));
    compute_graph_->SetGraphOutNodesInfo(output_nodes);
    GELOGI("********************SetOutputs Success***********************");
    GE_IF_BOOL_EXEC(!output_name_.empty(), GELOGI(" NetOutputs: (%s)", output_name_.c_str()));

    return GRAPH_SUCCESS;
  }

  graphStatus SetTargets(const std::vector<Operator> &targets) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Check][Param] set ComputeGraph faild.");
    GE_CHK_BOOL_EXEC_INFO(targets.size() != 0U, return GRAPH_SUCCESS, "set targets size is 0.");

    std::vector<ge::NodePtr> target_nodes;
    for (const auto item : targets) {
      const ge::NodePtr node = compute_graph_->FindNode(item.GetName());
      if (node == nullptr) {
        GELOGW("[SetTargets][Check] User designated target_node %s not exist in graph, skip it",
               item.GetName().c_str());
        continue;
      }
      target_nodes.push_back(node);
    }
    compute_graph_->SetGraphTargetNodesInfo(target_nodes);
    return GRAPH_SUCCESS;
  }
  bool IsValid() const { return (compute_graph_ != nullptr); }

  graphStatus AddOp(const ge::Operator &op) {
    const auto ret = op_list_.emplace(std::pair<std::string, ge::Operator>(op.GetName(), op));
    GE_CHK_BOOL_RET_STATUS(ret.second != false, GRAPH_FAILED, "[Check][Param] the op have added before, op name:%s.",
                           op.GetName().c_str());
    return GRAPH_SUCCESS;
  }

  graphStatus GetAllOpName(std::vector<std::string> &op_name) const {
    for (const auto &it : op_list_) {
      op_name.push_back(it.second.GetName());
    }
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByName(const std::string &name, ge::Operator &op) const {
    const auto it = op_list_.find(name);
    GE_CHK_BOOL_EXEC(it != op_list_.end(),
                     REPORT_INNER_ERROR("E19999", "there is no op: %s.", name.c_str());
                     return GRAPH_FAILED, "[Find][Op] there is no op: %s.", name.c_str());
    op = it->second;
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByType(const std::string &type, std::vector<ge::Operator> &ops) const {
    for (auto &op : op_list_) {
      auto op_type = op.second.GetOpType();
      if (op_type == type) {
        ops.push_back(op.second);
        continue;
      }
      if (op_type == ge::FRAMEWORKOP) {
        (void)op.second.GetAttr(ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op_type);
        if (op_type == type) {
          ops.push_back(op.second);
        }
      }
    }
    return GRAPH_SUCCESS;
  }

  void SetNeedIteration(bool need_iteration) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERROR("E19999", "Set need iteration failed, as compute graph is null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] Set need iteration failed, as compute graph is null.");
      return;
    }
    compute_graph_->SetNeedIteration(need_iteration);
  }

  const std::string &GetName() const {
    return name_;
  }

  ComputeGraphPtr GetComputeGraph() const {
    return compute_graph_;
  }

  graphStatus RemoveEdge(NodePtr &src_node_ptr, const int32_t src_port_index,
                         NodePtr &dst_node_ptr, const int32_t dst_port_index) {
    GE_CHECK_NOTNULL(src_node_ptr);
    GE_CHECK_NOTNULL(dst_node_ptr);

    graphStatus res = GRAPH_FAILED;
    if ((src_port_index == -1) && (dst_port_index == -1)) {
      if (src_node_ptr->GetOutControlAnchor() == nullptr) {
        REPORT_CALL_ERROR("E19999", "src node:%s out control anchor is null.", src_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][Anchor] src node:%s out control anchor is null.", src_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      res = GraphUtils::RemoveEdge(src_node_ptr->GetOutControlAnchor(), dst_node_ptr->GetInControlAnchor());
      if (res != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "remove control edge between [%s] and [%s]failed.",
                          src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Remove][ControlEdge] between [%s] and [%s]failed.",
               src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }

    if (src_node_ptr->GetOutDataAnchor(src_port_index) == nullptr) {
      REPORT_CALL_ERROR("E19999", "src node[%s] out data anchor[%d] is null.",
                        src_node_ptr->GetName().c_str(), src_port_index);
      GELOGE(GRAPH_FAILED, "[Get][Anchor] src node[%s] out data anchor[%d] is null.",
             src_node_ptr->GetName().c_str(), src_port_index);
      return GRAPH_FAILED;
    }

    if ((src_port_index != -1) && (dst_port_index == -1)) {
      res = GraphUtils::RemoveEdge(src_node_ptr->GetOutDataAnchor(src_port_index), dst_node_ptr->GetInControlAnchor());
      if (res != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "remove data-control edge between [%s] and [%s]failed.",
                          src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Remove][Edge] between [%s] and [%s]failed.",
               src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }

    res = GraphUtils::RemoveEdge(src_node_ptr->GetOutDataAnchor(src_port_index),
                                 dst_node_ptr->GetInDataAnchor(dst_port_index));
    if (res != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "remove data edge between [%s] and [%s] failed.",
                        src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][Edge] between [%s] and [%s] failed.",
             src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
  }

 private:
  std::string name_;
  std::string output_name_;
  std::map<std::string, ge::Operator> op_list_;
  ComputeGraphPtr compute_graph_{nullptr};
};

Graph::Graph(const std::string &name) {
  impl_ = ComGraphMakeShared<GraphImpl>(name);
  if (impl_ == nullptr) {
    GELOGW("[Check][Impl] Make graph impl failed");
  }
}

Graph::Graph(const char *name) {
  if (name != nullptr) {
    std::string graph_name = name;
    impl_ = ComGraphMakeShared<GraphImpl>(graph_name);
    if (impl_ == nullptr) {
      GELOGW("[Check][Impl] Make graph impl failed");
    }
  } else {
    GELOGW("[Check][Param] Input graph name is nullptr.");
  }
}

graphStatus Graph::AddOp(const ge::Operator &op) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] AddOp failed: graph can not be used, impl is nullptr.");
  return impl_->AddOp(op);
}

graphStatus Graph::GetAllOpName(std::vector<std::string> &op_name) const {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] GetAllOpName failed: graph can not be used, impl is nullptr.");
  return impl_->GetAllOpName(op_name);
}

graphStatus Graph::GetAllOpName(std::vector<AscendString> &names) const {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] GetAllOpName failed: graph can not be used, impl is nullptr.");
  std::vector<std::string> op_names;
  if (impl_->GetAllOpName(op_names) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Get][AllOpName] failed.");
    return GRAPH_FAILED;
  }

  for (auto &op_name : op_names) {
    names.emplace_back(op_name.c_str());
  }

  return GRAPH_SUCCESS;
}

graphStatus Graph::FindOpByName(const std::string &name, Operator &op) const {
  const Operator op_find_op_def("NULL");
  op = op_find_op_def;
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] FindOpByName failed: graph can not be used, impl is nullptr.");
  return impl_->FindOpByName(name, op);
}

graphStatus Graph::FindOpByName(const char *name, Operator &op) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] FindOpByName: name is nullptr.");
    return GRAPH_FAILED;
  }
  const Operator op_find_op_def("NULL");
  op = op_find_op_def;
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] FindOpByName failed: graph can not be used, impl is nullptr.");
  const std::string op_name = name;
  return impl_->FindOpByName(op_name, op);
}

graphStatus Graph::FindOpByType(const std::string &type, std::vector<ge::Operator> &ops) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->FindOpByType(type, ops);
}

graphStatus Graph::FindOpByType(const char *type, std::vector<ge::Operator> &ops) const {
  if (type == nullptr) {
    REPORT_INNER_ERROR("E19999", "param type is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] FindOpByType: type is nullptr.");
    return GRAPH_FAILED;
  }
  GE_CHECK_NOTNULL(impl_);
  const std::string op_type = type;
  return impl_->FindOpByType(op_type, ops);
}

Graph &Graph::SetInputs(const std::vector<ge::Operator> &inputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetInputs failed: graph can not be used, impl is nullptr.");
  GE_CHK_BOOL_EXEC(inputs.size() > 0U, REPORT_INNER_ERROR("E19999", "input operator size can not be 0");
                   return *this, "[Check][Param] SetInputs failed: input operator size can not be 0.");
  (void)impl_->SetInputs(inputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<ge::Operator> &outputs) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(outputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(output_indexs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, std::string>> &outputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.")
  (void)impl_->SetOutputs(outputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<ge::Operator, AscendString>> &outputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.")
  std::vector<std::pair<ge::Operator, std::string>> graph_outputs;
  for (auto &item : outputs) {
    const char * const name = item.second.GetString();
    if (name != nullptr) {
      graph_outputs.emplace_back((std::pair<ge::Operator, std::string>(item.first, name)));
    } else {
      GELOGW("[SetOutputs][CheckParam] Input output_op_name is nullptr.");
    }
  }

  (void)impl_->SetOutputs(graph_outputs);
  return *this;
}

Graph &Graph::SetTargets(const std::vector<ge::Operator> &targets) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetTargets failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetTargets(targets);
  return *this;
}

bool Graph::IsValid() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->IsValid();
}

void Graph::SetNeedIteration(bool need_iteration) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Set need iteration failed, as impl is null.");
    return;
  }
  impl_->SetNeedIteration(need_iteration);
}

std::vector<GNode> Graph::GetAllNodes() const {
  std::vector<GNode> graph_nodes;
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAllNodes: graph can not be used, impl is nullptr.");
    return graph_nodes;
  }

  const ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  if (compute_graph_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "impl compute graph is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][Graph] GetAllNodes: compute graph ptr is nullptr.");
    return graph_nodes;
  }

  for (auto &node : compute_graph_ptr->GetAllNodes()) {
    GNode gnode = NodeAdapter::Node2GNode(node);
    graph_nodes.emplace_back(gnode);
  }

  return graph_nodes;
}

std::vector<GNode> Graph::GetDirectNode() const {
  std::vector<GNode> graph_nodes;
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetDirectNode: graph can not be used, impl is nullptr.");
    return graph_nodes;
  }
  const ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  if (compute_graph_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "impl compute graph is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][Graph] GetDirectNode: compute graph ptr is nullptr.");
    return graph_nodes;
  }

  for (auto &node : compute_graph_ptr->GetDirectNode()) {
    GNode gnode = NodeAdapter::Node2GNode(node);
    graph_nodes.emplace_back(gnode);
  }

  return graph_nodes;
}

graphStatus Graph::RemoveNode(GNode &node) {
  return RemoveNode(node, false);
}

graphStatus Graph::RemoveNode(GNode &node, bool contain_subgraph) {
  GE_CHECK_NOTNULL(impl_);

  const NodePtr node_ptr = NodeAdapter::GNode2Node(node);
  GE_CHECK_NOTNULL(node_ptr);

  const ComputeGraphPtr owner_compute_graph = node_ptr->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_compute_graph);

  ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  GE_CHECK_NOTNULL(compute_graph_ptr);

  if (contain_subgraph) {
    if (!GraphUtils::IsNodeInGraphRecursively(compute_graph_ptr, *node_ptr)) {
      REPORT_CALL_ERROR("E19999", "node[%s] is not in the graph[%s] or not in subgraph.",
                        node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] is not in the graph[%s].",
             node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    compute_graph_ptr = owner_compute_graph;
  } else {
    if (compute_graph_ptr != owner_compute_graph) {
      REPORT_INNER_ERROR("E19999", "node[%s] is not in the graph[%s].",
                         node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] is not in the graph[%s].",
             node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  ge::NodeUtils::UnlinkAll(*node_ptr);
  if (GraphUtils::RemoveNodeWithoutRelink(compute_graph_ptr, node_ptr) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "graph:%s remove node:%s failed",
                      compute_graph_ptr->GetName().c_str(), node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Remove][Node] %s from graph:%s failed.",
           node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  (void)node_ptr->ClearOwnerGraph(nullptr);
  return GRAPH_SUCCESS;
}

graphStatus Graph::RemoveEdge(GNode &src_node, const int32_t src_port_index,
                              GNode &dst_node, const int32_t dst_port_index) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  if ((src_port_index == -1) && (dst_port_index != -1)) {
    REPORT_INNER_ERROR("E19999", "src_port_index == -1 and dst_port_index != -1, check invalid .");
    GELOGE(GRAPH_FAILED, "[Check][Param] src control anchor link to dst data anchor not exists.");
    return GRAPH_FAILED;
  }

  NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "src node:%s compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node:%s compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst node:%s compute graph is nullptr", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node:%s compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (impl_->RemoveEdge(src_node_ptr, src_port_index, dst_node_ptr, dst_port_index) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "remove edge between %s(%d) and %s(%d) failed.",
                      src_node_ptr->GetName().c_str(), src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    GELOGE(GRAPH_FAILED, "[Remove][Edge] between %s(%d) and %s(%d) failed.",
           src_node_ptr->GetName().c_str(), src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GNode Graph::AddNodeByOp(const Operator &op) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GNode();
  }

  const std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "get op desc from op:%s failed", op.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] from op[%s] failed.", op.GetName().c_str());
    return  GNode();
  }

  const ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  if (compute_graph_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "impl compute graph is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][Graph] compute graph ptr is nullptr.");
    return GNode();
  }

  const NodePtr node_ptr = compute_graph_ptr->AddNode(op_desc);
  const GNode gnode = NodeAdapter::Node2GNode(node_ptr);

  return gnode;
}

graphStatus Graph::AddDataEdge(GNode &src_node, const int32_t src_port_index,
                               GNode &dst_node, const int32_t dst_port_index) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  const NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  const NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const graphStatus res = GraphUtils::AddEdge(src_node_ptr->GetOutDataAnchor(src_port_index),
                                              dst_node_ptr->GetInDataAnchor(dst_port_index));
  if (res != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add data edge from %s(%d) to %s(%d) failed.", src_node_ptr->GetName().c_str(),
                      src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    GELOGE(GRAPH_FAILED, "[Add][DataEdge] from %s(%d) to %s(%d) failed.", src_node_ptr->GetName().c_str(),
           src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus Graph::AddControlEdge(GNode &src_node, GNode &dst_node) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  const NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  const NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_CALL_ERROR("E19999", "dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const graphStatus res = GraphUtils::AddEdge(src_node_ptr->GetOutControlAnchor(), dst_node_ptr->GetInControlAnchor());
  if (res != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add control edge from %s to %s failed.", src_node_ptr->GetName().c_str(),
                      dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed.", src_node_ptr->GetName().c_str(),
           dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return SUCCESS;
}

GraphPtr Graph::ConstructFromInputs(const std::vector<Operator> &inputs, const AscendString &name) {
  const char* ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "ascend string error");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] ascend string error.");
    return nullptr;
  }

  if (inputs.empty()) {
    REPORT_INNER_ERROR("E19999", "inputs size can not be 0.");
    GELOGE(GRAPH_FAILED, "[Check][Param] inputs size can not be 0.");
    return nullptr;
  }

  std::string graph_name = ascend_name;
  ComputeGraphPtr compute_graph = GraphUtils::CreateGraphFromOperator(graph_name, inputs);
  if (compute_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "create compute graph from op failed, name:%s", graph_name.c_str());
    GELOGE(GRAPH_FAILED, "[Create][ComputeGraph] failed, name:%s.", graph_name.c_str());
    return nullptr;
  }

  compute_graph->SetInputSize(static_cast<uint32_t>(inputs.size()));
  GraphPtr graph_ptr = GraphUtils::CreateGraphPtrFromComputeGraph(compute_graph);
  if (graph_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "create graph from compute graph:%s failed.", compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Create][Graph] from compute graph:%s failed.", compute_graph->GetName().c_str());
    return nullptr;
  }

  return graph_ptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtils::GetComputeGraph(const ge::Graph &graph) {
  GE_CHK_BOOL_EXEC_NOLOG(graph.IsValid(), return nullptr);
  return graph.impl_->compute_graph_;
}

graphStatus Graph::SaveToFile(const std::string &file_name) const {
  Model model = Model();
  model.SetGraph(*this);
  return model.SaveToFile(file_name);
}

graphStatus Graph::SaveToFile(const char *file_name) const {
  if (file_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "file name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] file name is nullptr.");
    return GRAPH_FAILED;
  }

  Model model = Model();
  model.SetGraph(*this);
  std::string file = file_name;
  return model.SaveToFile(file);
}

graphStatus Graph::LoadFromFile(const std::string &file_name) {
  Model model = Model();
  graphStatus ret = model.LoadFromFile(file_name);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  *this = model.GetGraph();
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromFile(const char *file_name) {
  if (file_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param file name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] file name is nullptr.");
    return GRAPH_FAILED;
  }

  Model model = Model();
  const std::string file = file_name;
  const graphStatus ret = model.LoadFromFile(file);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  *this = model.GetGraph();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::string &Graph::GetName() const {
  return impl_->GetName();
}

graphStatus Graph::GetName(AscendString &name) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] impl is nullptr.");
    return GRAPH_FAILED;
  }
  const std::string graph_name = impl_->GetName();
  name = AscendString(graph_name.c_str());
  return GRAPH_SUCCESS;
}

graphStatus Graph::CopyFrom(const Graph &src_graph) {
  const auto res = GraphUtils::CopyGraph(src_graph, *this);
  if (res != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "copy graph from %s failed.", src_graph.GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Copy][Graph] from %s failed.", src_graph.GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph
GraphUtils::CreateGraphFromComputeGraph(const ge::ComputeGraphPtr compute_graph) {
  GE_CHK_BOOL_EXEC_NOLOG(compute_graph != nullptr, return Graph(""));

  const auto name = compute_graph->GetName();
  const auto graph = Graph(name);

  GE_CHK_BOOL_EXEC_NOLOG(graph.impl_ != nullptr, return graph);
  graph.impl_->compute_graph_ = compute_graph;

  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::CopyGraphImpl(const Graph &src_graph, Graph &dst_graph,
                          const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                          const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new) {
  GE_CHECK_NOTNULL(dst_graph.impl_);
  GE_CHECK_NOTNULL(src_graph.impl_);

  std::map<std::string, ge::Operator> &dst_op_list = dst_graph.impl_->op_list_;
  const std::map<std::string, ge::Operator> &src_op_list = src_graph.impl_->op_list_;
  auto &dst_compute_graph = dst_graph.impl_->compute_graph_;

  dst_graph.impl_->output_name_ = src_graph.impl_->output_name_;

  auto ret = OpDescUtils::CopyOperators(dst_compute_graph,
                                        node_old_2_new, op_desc_old_2_new,
                                        src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "copy operators to graph:%s failed.", dst_compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Copy][Operators] to graph:%s failed.", dst_compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }

  ret = OpDescUtils::CopyOperatorLinks(src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "copy operator links failed, ret:%d.", ret);
    GELOGE(GRAPH_FAILED, "[Copy][OperatorLinks] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GraphPtr
GraphUtils::CreateGraphPtrFromComputeGraph(const ge::ComputeGraphPtr compute_graph) {
  GE_CHK_BOOL_EXEC_NOLOG(compute_graph != nullptr, return nullptr);

  auto name = compute_graph->GetName();
  const auto graph = ComGraphMakeShared<Graph>(name);
  GE_CHK_BOOL_EXEC_NOLOG(graph != nullptr, return nullptr);
  GE_CHK_BOOL_EXEC_NOLOG(graph->impl_ != nullptr, return nullptr);

  graph->impl_->compute_graph_ = compute_graph;

  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::GetIndependentCompileGraphs(const ComputeGraphPtr &compute_graph,
                                                    std::vector<ComputeGraphPtr> &independent_compile_subgraphs) {
  bool is_pipeline_partitioned = false;
  (void)AttrUtils::GetBool(*compute_graph, ATTR_NAME_PIPELINE_PARTITIONED, is_pipeline_partitioned);
  if (is_pipeline_partitioned) {
    for (const auto &node : compute_graph->GetDirectNode()) {
      if (node->GetType() == PARTITIONEDCALL) {
        auto sub_graph = NodeUtils::GetSubgraph(*node, kSubgraphIndexOfPartitionedCall);
        GE_CHECK_NOTNULL(sub_graph);
        independent_compile_subgraphs.emplace_back(sub_graph);
      }
    }
    return GRAPH_SUCCESS;
  }
  independent_compile_subgraphs.emplace_back(compute_graph);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::RecoverGraphOperators(const Graph &graph) {
  GE_CHECK_NOTNULL(graph.impl_);
  GE_CHECK_NOTNULL(graph.impl_->compute_graph_);

  graph.impl_->op_list_.clear();
  for (const auto &node : graph.impl_->compute_graph_->GetDirectNode()) {
    graph.impl_->op_list_[node->GetName()] = OpDescUtils::CreateOperatorFromNode(node);
  }
  return SUCCESS;
}
}  // namespace ge
