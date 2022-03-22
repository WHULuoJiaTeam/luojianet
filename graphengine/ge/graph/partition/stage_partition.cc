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

#include "graph/partition/stage_partition.h"

#include <stack>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/common/util.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const std::set<std::string> kSrcNodeTypes = { DATA, AIPPDATA, ANN_DATA };
}

Status StagePartitioner::Partition() {
  GE_CHECK_NOTNULL(root_graph_);
  if (root_graph_->GetParentGraph() != nullptr) {
    return SUCCESS;
  }

  for (const auto &node : root_graph_->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    uint32_t level = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, level)) {
      continue;
    }
    if ((kSrcNodeTypes.count(op_desc->GetType()) != 0) && node->GetInAllNodes().empty()) {
      continue;
    }
    GELOGD("original node %s for stage %u", node->GetName().c_str(), level);
    stage_nodes_[level].insert(node);
  }
  if (stage_nodes_.empty()) {
    GELOGI("Graph %s does not set stage_level, it is not_changed.", root_graph_->GetName().c_str());
    return SUCCESS;
  }

  GE_DUMP(root_graph_, "BeforeStagePartition");
  if (SplitStageLevel() != SUCCESS) {
    GELOGE(FAILED, "[Split][GraphStage] for graph %s failed.", root_graph_->GetName().c_str());
    return FAILED;
  }

  if (StagePartition() != SUCCESS) {
    GELOGE(FAILED, "[Stage][Partition] for graph %s failed.", root_graph_->GetName().c_str());
    return FAILED;
  }

  root_graph_->TopologicalSorting([](const NodePtr &a, const NodePtr &b) -> bool {
    uint32_t a_level = UINT32_MAX;
    (void)AttrUtils::GetInt(a->GetOpDesc(), ATTR_STAGE_LEVEL, a_level);
    uint32_t b_level = UINT32_MAX;
    (void)AttrUtils::GetInt(b->GetOpDesc(), ATTR_STAGE_LEVEL, b_level);
    return a_level < b_level;
  });
  if (root_graph_->TopologicalSorting() != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Call][TopologicalSorting] for graph %s after stage partition failed, "
           "maybe stage_level was not set correctly.", root_graph_->GetName().c_str());
    return FAILED;
  }
  GE_DUMP(root_graph_, "AfterStagePartition");
  return SUCCESS;
}

Status StagePartitioner::SplitStageLevel() {
  std::stack<NodePtr> nodes;
  std::unordered_set<NodePtr> visited_stage_nodes;
  for (auto &stage : stage_nodes_) {
    uint32_t cur_stage_level = stage.first;
    const auto &cur_stage_nodes = stage.second;
    for (const auto &marked_node : cur_stage_nodes) {
      nodes.push(marked_node);
    }
    visited_stage_nodes.clear();
    while (!nodes.empty()) {
      auto node = nodes.top();
      nodes.pop();
      GE_CHECK_NOTNULL(node->GetOpDesc());
      uint32_t tmp_level = cur_stage_level;
      (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_STAGE_LEVEL, tmp_level);
      if (tmp_level != cur_stage_level) {
        continue;
      }
      for (const auto &in_node : node->GetInAllNodes()) {
        if (visited_stage_nodes.count(in_node) != 0) {
          continue;
        }
        if (!AttrUtils::SetInt(in_node->GetOpDesc(), ATTR_STAGE_LEVEL, cur_stage_level)) {
          REPORT_CALL_ERROR("E19999", "Set Attr %s on node %s failed.",
                            ATTR_STAGE_LEVEL.c_str(), in_node->GetName().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Attr] %s on node %s failed.",
                 ATTR_STAGE_LEVEL.c_str(), in_node->GetName().c_str());
          return INTERNAL_ERROR;
        }
        GELOGD("Mark stage_level node %s, stage_level=%u", in_node->GetName().c_str(), cur_stage_level);
        if ((kSrcNodeTypes.count(in_node->GetType()) != 0) && in_node->GetInAllNodes().empty()) {
          GELOGD("skip data node %s for stage %u", in_node->GetName().c_str(), cur_stage_level);
          continue;
        }
        nodes.push(in_node);
      }
      visited_stage_nodes.emplace(node);
    }
    for (const auto &node : visited_stage_nodes) {
      stage.second.insert(node);
    }
  }

  return SUCCESS;
}

Status StagePartitioner::StagePartition() {
  for (const auto &stage : stage_nodes_) {
    StageInfo stage_info(stage.first);
    FindStageIO(stage.second, stage_info);

    std::string subgraph_name = "Subgraph_Level_" + std::to_string(stage.first);
    NodePtr graph_node = BuildSubgraphNode(subgraph_name, stage_info);
    if (graph_node == nullptr) {
      GELOGE(FAILED, "[Build][SubgraphNode] for stage %u failed, graph name:%s.", stage.first, subgraph_name.c_str());
      return FAILED;
    }

    ComputeGraphPtr subgraph = BuildStageGraph(graph_node, stage_info);
    if (subgraph == nullptr) {
      GELOGE(FAILED, "[Build][StageGraph] %s for stage %u failed.", graph_node->GetName().c_str(), stage.first);
      return FAILED;
    }
    if (root_graph_->AddSubgraph(subgraph) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "add subgraph:%s in root graph:%s of stage %u failed.",
                        subgraph->GetName().c_str(), root_graph_->GetName().c_str(), stage.first);
      GELOGE(FAILED, "[Add][SubGraph] %s in root graph:%s of stage %u failed.",
             subgraph->GetName().c_str(), root_graph_->GetName().c_str(), stage.first);
      return FAILED;
    }

    if ((RelinkDataEdges(graph_node, stage_info) != SUCCESS) ||
        (RelinkCtrlEdges(graph_node, stage_info) != SUCCESS)) {
      GELOGE(FAILED, "[ReLink][Edges] for stage %u failed, graph_node:%s.", stage.first, graph_node->GetName().c_str());
      return FAILED;
    }

    for (const auto &stage_node : stage.second) {
      if (GraphUtils::RemoveNodeWithoutRelink(root_graph_, stage_node) != GRAPH_SUCCESS) {
        GELOGW("Remove node %s failed.", stage_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

void StagePartitioner::FindStageIO(const std::unordered_set<NodePtr> &stage_nodes, StageInfo &stage_info) {
  for (const auto &node : stage_nodes) {
    // stage nodes
    stage_info.stage_nodes.emplace(node);
    // in data nodes
    for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
      OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      if (stage_nodes.count(peer_out_anchor->GetOwnerNode()) == 0) {
        stage_info.data_inputs.emplace_back(std::make_pair(peer_out_anchor, in_data_anchor));
      } else {
        stage_info.inner_data_edges.emplace_back(std::make_pair(peer_out_anchor, in_data_anchor));
      }
    }
    // out data nodes
    std::list<InDataAnchorPtr> peer_data_anchors;
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      peer_data_anchors.clear();
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        if (stage_nodes.count(peer_in_anchor->GetOwnerNode()) == 0) {
          peer_data_anchors.emplace_back(peer_in_anchor);
        }
      }
      if (!peer_data_anchors.empty()) {
        stage_info.data_outputs.emplace_back(std::make_pair(out_data_anchor, peer_data_anchors));
      }
    }
    // in ctrl nodes
    for (const auto &in_ctrl_node : node->GetInControlNodes()) {
      if (stage_nodes.count(in_ctrl_node) == 0) {
        stage_info.ctrl_inputs.emplace_back(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor());
      } else {
        stage_info.inner_ctrl_edges.emplace_back(std::make_pair(in_ctrl_node->GetOutControlAnchor(),
                                                                node->GetInControlAnchor()));
      }
    }
    // out ctrl nodes
    for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (stage_nodes.count(out_ctrl_node) == 0) {
        stage_info.ctrl_outputs.emplace_back(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor());
      }
    }
  }
}

NodePtr StagePartitioner::BuildSubgraphNode(const std::string &graph_name, const StageInfo &stage_info) {
  OpDescBuilder op_desc_builder(graph_name, PARTITIONEDCALL);
  size_t input_num = stage_info.data_inputs.size();
  for (size_t i = 0; i < input_num; i++) {
    auto input_desc = stage_info.data_inputs[i].second->GetOwnerNode()->GetOpDesc();
    if (input_desc == nullptr) {
      GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
             stage_info.data_inputs[i].second->GetOwnerNode()->GetName().c_str());
      return nullptr;
    }
    op_desc_builder.AddInput("args" + std::to_string(i),
                             input_desc->GetInputDesc(stage_info.data_inputs[i].second->GetIdx()));
  }
  size_t output_num = stage_info.data_outputs.size();
  for (size_t i = 0; i < output_num; i++) {
    auto output_desc = stage_info.data_outputs[i].first->GetOwnerNode()->GetOpDesc();
    if (output_desc == nullptr) {
      GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
             stage_info.data_outputs[i].first->GetOwnerNode()->GetName().c_str());
      return nullptr;
    }
    op_desc_builder.AddOutput("output" + std::to_string(i),
                              output_desc->GetOutputDesc(stage_info.data_outputs[i].first->GetIdx()));
  }

  OpDescPtr op_desc = op_desc_builder.Build();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "[Create][OpDesc] for subgraph node failed, name:%s.", graph_name.c_str());
    return nullptr;
  }

  op_desc->AddSubgraphName("f");
  op_desc->SetSubgraphInstanceName(0, graph_name);

  if (!AttrUtils::SetInt(op_desc, ATTR_STAGE_LEVEL, stage_info.stage_level)) {
    REPORT_CALL_ERROR("E19999", "set attr %s on node %s failed", ATTR_STAGE_LEVEL.c_str(), op_desc->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s on node %s failed", ATTR_STAGE_LEVEL.c_str(), op_desc->GetName().c_str());
    return nullptr;
  }

  NodePtr subgraph_node = root_graph_->AddNode(op_desc);
  if (subgraph_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "add node:%s in graph:%s failed.",
                      op_desc->GetName().c_str(), root_graph_->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s in graph:%s failed.", op_desc->GetName().c_str(), root_graph_->GetName().c_str());
    return nullptr;
  }
  if (subgraph_node->SetOwnerComputeGraph(root_graph_) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetOwnerComputeGraph for node %s failed, grpah:%s.",
                      subgraph_node->GetName().c_str(), root_graph_->GetName().c_str());
    GELOGE(FAILED, "[Set][OwnerGraph] for node %s failed, grpah:%s.",
           subgraph_node->GetName().c_str(), root_graph_->GetName().c_str());
    return nullptr;
  }

  return subgraph_node;
}

ComputeGraphPtr StagePartitioner::BuildStageGraph(const NodePtr &subgraph_node, const StageInfo &stage_info) {
  CompleteGraphBuilder graph_builder(subgraph_node->GetName(), false);
  // Add parent node
  graph_builder.SetParentNode(subgraph_node);

  // Add node
  for (const auto &node : stage_info.stage_nodes) {
    graph_builder.AddNode(AttrUtils::CopyOpDesc(node->GetOpDesc()));
  }

  // Set Input
  size_t data_input_num = stage_info.data_inputs.size();
  for (size_t i = 0; i < data_input_num; i++) {
    graph_builder.SetInput(i, { stage_info.data_inputs[i].second->GetOwnerNode()->GetName() },
                           { static_cast<uint32_t>(stage_info.data_inputs[i].second->GetIdx()) });
  }

  // Add Outputs
  size_t data_output_num = stage_info.data_outputs.size();
  for (uint32_t i = 0; i < data_output_num; i++) {
    graph_builder.AddOutput(stage_info.data_outputs[i].first->GetOwnerNode()->GetName(),
                            stage_info.data_outputs[i].first->GetIdx());
  }

  // Add Data Edges
  for (const auto &data_edge : stage_info.inner_data_edges) {
    graph_builder.AddDataLink(data_edge.first->GetOwnerNode()->GetName(), data_edge.first->GetIdx(),
                              data_edge.second->GetOwnerNode()->GetName(), data_edge.second->GetIdx());
  }

  // Add Ctrl Edges
  for (const auto &ctrl_edge : stage_info.inner_ctrl_edges) {
    graph_builder.AddControlLink(ctrl_edge.first->GetOwnerNode()->GetName(),
                                 ctrl_edge.second->GetOwnerNode()->GetName());
  }

  // Add Input-Mapping
  std::map<uint32_t, uint32_t> input_mapping;
  for (size_t i = 0; i < data_input_num; i++) {
    input_mapping[i] = i;
  }
  graph_builder.SetInputMapping(input_mapping);

  // Add outputMapping
  std::map<uint32_t, uint32_t> output_mapping;
  for (size_t i = 0; i < data_output_num; i++) {
    output_mapping[i] = i;
  }
  graph_builder.SetOutputMapping(output_mapping);

  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr subgraph = graph_builder.Build(error_code, error_msg);
  if (subgraph == nullptr) {
    GELOGE(error_code, "[Build][Subgraph] %s failed:%s.", subgraph_node->GetName().c_str(), error_msg.c_str());
    return nullptr;
  }
  if (!AttrUtils::SetInt(subgraph, ATTR_STAGE_LEVEL, stage_info.stage_level)) {
    REPORT_CALL_ERROR("E19999", "set attr %s on graph %s failed.",
                      ATTR_STAGE_LEVEL.c_str(), subgraph->GetName().c_str());
    GELOGE(FAILED, "[Set][Attr] %s on graph %s failed.", ATTR_STAGE_LEVEL.c_str(), subgraph->GetName().c_str());
    return nullptr;
  }

  return subgraph;
}

Status StagePartitioner::RelinkDataEdges(const NodePtr &subgraph_node, const StageInfo &stage_info) {
  // in data nodes
  for (size_t i = 0; i < stage_info.data_inputs.size(); i++) {
    if (stage_info.data_inputs[i].first->Unlink(stage_info.data_inputs[i].second) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "remove data edge from %s:%d to %s:%d failed",
                        stage_info.data_inputs[i].first->GetOwnerNode()->GetName().c_str(),
                        stage_info.data_inputs[i].first->GetIdx(),
                        stage_info.data_inputs[i].second->GetOwnerNode()->GetName().c_str(),
                        stage_info.data_inputs[i].second->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Remove][DataEdge] %s:%d->%s:%d failed.",
             stage_info.data_inputs[i].first->GetOwnerNode()->GetName().c_str(),
             stage_info.data_inputs[i].first->GetIdx(),
             stage_info.data_inputs[i].second->GetOwnerNode()->GetName().c_str(),
             stage_info.data_inputs[i].second->GetIdx());
      return INTERNAL_ERROR;
    }
    if (stage_info.data_inputs[i].first->LinkTo(subgraph_node->GetInDataAnchor(i)) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "add data edge from %s:%d to %s:%zu failed.",
                        stage_info.data_inputs[i].first->GetOwnerNode()->GetName().c_str(),
                        stage_info.data_inputs[i].first->GetIdx(),
                        subgraph_node->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[Add][DataEdge] %s:%d->%s:%zu failed.",
             stage_info.data_inputs[i].first->GetOwnerNode()->GetName().c_str(),
             stage_info.data_inputs[i].first->GetIdx(),
             subgraph_node->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }
  }
  // out data nodes
  for (size_t i = 0; i < stage_info.data_outputs.size(); i++) {
    const auto &out_data_anchor = subgraph_node->GetOutDataAnchor(i);
    GE_CHECK_NOTNULL(out_data_anchor);
    for (const auto &peer_in_anchor : stage_info.data_outputs[i].second) {
      if (stage_info.data_outputs[i].first->Unlink(peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Remove data edge from %s:%d to %s:%d failed.",
                          stage_info.data_outputs[i].first->GetOwnerNode()->GetName().c_str(),
                          stage_info.data_outputs[i].first->GetIdx(),
                          peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Remove][DataEdge] %s:%d->%s:%d failed.",
               stage_info.data_outputs[i].first->GetOwnerNode()->GetName().c_str(),
               stage_info.data_outputs[i].first->GetIdx(),
               peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
        return INTERNAL_ERROR;
      }
      if (out_data_anchor->LinkTo(peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add data edge from %s:%zu to %s:%d failed.", subgraph_node->GetName().c_str(), i,
                          peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Add][DataEdge] %s:%zu->%s:%d failed.", subgraph_node->GetName().c_str(), i,
               peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status StagePartitioner::RelinkCtrlEdges(const NodePtr &subgraph_node, const StageInfo &stage_info) {
  // in ctrl nodes
  for (const auto &ctrl_input : stage_info.ctrl_inputs) {
    if (ctrl_input.first->Unlink(ctrl_input.second) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove ctrl edge %s->%s failed.",
                        ctrl_input.first->GetOwnerNode()->GetName().c_str(),
                        ctrl_input.second->GetOwnerNode()->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Remove][CtrlEdge] %s->%s failed.",
             ctrl_input.first->GetOwnerNode()->GetName().c_str(), ctrl_input.second->GetOwnerNode()->GetName().c_str());
      return INTERNAL_ERROR;
    }
    if (!ctrl_input.first->IsLinkedWith(subgraph_node->GetInControlAnchor())) {
      if (ctrl_input.first->LinkTo(subgraph_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add ctrl edge %s->%s failed.",
                          ctrl_input.first->GetOwnerNode()->GetName().c_str(), subgraph_node->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] %s->%s failed.",
               ctrl_input.first->GetOwnerNode()->GetName().c_str(), subgraph_node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  // out ctrl nodes
  for (const auto &ctrl_output : stage_info.ctrl_outputs) {
    if (ctrl_output.first->Unlink(ctrl_output.second) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove ctrl edge %s->%s failed.",
                        ctrl_output.first->GetOwnerNode()->GetName().c_str(),
                        ctrl_output.second->GetOwnerNode()->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Remove][CtrlEdge] %s->%s failed.",
             ctrl_output.first->GetOwnerNode()->GetName().c_str(),
             ctrl_output.second->GetOwnerNode()->GetName().c_str());
      return INTERNAL_ERROR;
    }
    if (!subgraph_node->GetOutControlAnchor()->IsLinkedWith(ctrl_output.second)) {
      if (subgraph_node->GetOutControlAnchor()->LinkTo(ctrl_output.second) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add ctrl edge %s->%s failed.",
                          subgraph_node->GetName().c_str(), ctrl_output.second->GetOwnerNode()->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] %s->%s failed.",
               subgraph_node->GetName().c_str(), ctrl_output.second->GetOwnerNode()->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}
}  // namespace ge
