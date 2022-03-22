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

#include "graph/passes/multi_batch_pass.h"

#include <stack>
#include <unordered_set>
#include "common/ge/ge_util.h"
#include "common/omg_util.h"
#include "graph/utils/type_utils.h"
#include "common/formats/utils/formats_trans_utils.h"

namespace ge {
Status MultiBatchPass::Run(ComputeGraphPtr graph) {
  GELOGD("MultiBatchPass Enter");

  if (graph->GetParentGraph() != nullptr) {
    GELOGI("Subgraph %s skip the MultiBatchPass.", graph->GetName().c_str());
    return SUCCESS;
  }
  OutDataAnchorPtr pred_value = nullptr;
  Status ret = FindPredValue(graph, pred_value);
  if (ret == NOT_CHANGED) {
    GELOGD("SwitchN node not exist, graph not changed.");
    return SUCCESS;
  }
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Find][PredValue] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }

  if (GetDynamicType() != SUCCESS) {
    GELOGE(FAILED, "[Get][DynamicType] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }
  if (GetUserDesignateShape() != SUCCESS) {
    GELOGE(FAILED, "[Get][UserDesignateShape] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }
  std::vector<std::vector<int64_t>> batch_shape;
  std::vector<std::vector<int64_t>> combined_batch;
  if (!CheckSwitchN(batch_shape, combined_batch)) {
    GELOGE(FAILED, "[Check][SwitchN] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }

  if (attach_label_only_) {
    return AttachLabelOnly(batch_shape.size());
  }

  if (FindSwitchOutNodes(batch_shape.size()) != SUCCESS) {
    GELOGE(FAILED, "[Find][SwitchOutNodes] in graph:%s failed, batch_num:%zu.",
           graph->GetName().c_str(), batch_shape.size());
    return FAILED;
  }

  if (ReplaceSwitchN(graph, pred_value, batch_shape, combined_batch) != SUCCESS) {
    GELOGE(FAILED, "[Replace][SwitchN] in graph:%s failed.", graph->GetName().c_str());
    return FAILED;
  }

  for (const NodePtr &node : bypass_nodes_) {
    if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      GELOGE(FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
             node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      return FAILED;
    }
  }

  GELOGD("MultiBatchPass Leave");
  return SUCCESS;
}

///
/// @brief Clear Status
/// @return
///
Status MultiBatchPass::ClearStatus() {
  switch_n_nodes_.clear();
  bypass_nodes_.clear();
  batch_head_nodes_.clear();
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set batch label for Case mode.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] const NodePtr &case_node: Case Node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchPass::SetCaseLabel(const ComputeGraphPtr &graph, const NodePtr &case_node) {
  const auto &func_desc = case_node->GetOpDesc();
  GE_CHECK_NOTNULL(func_desc);
  if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
    GELOGD("Graph: %s Not multi-batch, Node: %s", graph->GetName().c_str(), case_node->GetName().c_str());
    return SUCCESS;
  }

  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < dynamic_branch_names.size(); ++i) {
    const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[i]);
    GE_CHECK_NOTNULL(subgraph);

    const std::string batch_label = "Batch_" + std::to_string(i);
    for (const auto &node : subgraph->GetDirectNode()) {
      (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
    }
  }

  return SUCCESS;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [out] pred_value
/// @return Status
///
Status MultiBatchPass::FindPredValue(const ComputeGraphPtr &graph, OutDataAnchorPtr &pred_value) {
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == CASE) {
      GE_CHK_STATUS_RET(SetCaseLabel(graph, node),
                        "[Set][CaseLabel] for node:%s(%s) in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      continue;
    }
    if (node->GetType() != SWITCHN) {
      continue;
    }

    const auto &in_data_anchor = node->GetInDataAnchor(SWITCH_PRED_INPUT);
    if (in_data_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Index:%u data anchor of node:%s(%s) is nullptr, check invalid",
                         SWITCH_PRED_INPUT, node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Get][InDataAnchor] failed, Index:%u data anchor of node:%s(%s) is nullptr.",
             SWITCH_PRED_INPUT, node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
    const auto &pred_input = in_data_anchor->GetPeerOutAnchor();
    if (pred_input == nullptr) {
      REPORT_INNER_ERROR("E19999", "Index:%u data anchor of node:%s(%s), its peer anchor is nullptr, check invalid",
                         SWITCH_PRED_INPUT, node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Get][PeerOutAnchor] failed, Index:%u data anchor of node:%s(%s), its peer anchor is nullptr.",
             SWITCH_PRED_INPUT, node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }

    if (pred_value == nullptr) {
      pred_value = pred_input;
    } else if (pred_value != pred_input) {
      REPORT_INNER_ERROR("E19999", "Multi pred_value of case node exist in graph:%s, check invalid",
                         graph->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Multi pred_value of case node exist in graph:%s.", graph->GetName().c_str());
      return FAILED;
    }
    switch_n_nodes_.emplace_back(node);
  }

  if (switch_n_nodes_.empty()) {
    GELOGD("SwitchN node not exist.");
    return NOT_CHANGED;
  }

  if (pred_value == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find Pred Input of case node in graph:%s failed", graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] FindPredInput in graph:%s failed, pred_value is null.", graph->GetName().c_str());
    return FAILED;
  }

  GELOGI("Find pred_value %s.", pred_value->GetOwnerNode()->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Get dynamic type: dynamic batch size: 1, dynamic image size: 2, dynamic dims: 3
/// @return Status
///
Status MultiBatchPass::GetDynamicType() {
  for (const auto &switch_n : switch_n_nodes_) {
    int32_t dynamic_type = static_cast<int32_t>(FIXED);
    if (!AttrUtils::GetInt(switch_n->GetOpDesc(), ATTR_DYNAMIC_TYPE, dynamic_type)) {
      REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_DYNAMIC_TYPE.c_str(),
                        switch_n->GetName().c_str(), switch_n->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_DYNAMIC_TYPE.c_str(),
             switch_n->GetName().c_str(), switch_n->GetType().c_str());
      return FAILED;
    }
    if (dynamic_type == static_cast<int32_t>(FIXED)) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%d check invalid", ATTR_DYNAMIC_TYPE.c_str(),
                         switch_n->GetName().c_str(), switch_n->GetType().c_str(), dynamic_type);
      GELOGE(FAILED, "[Check][Param] Attr:%s in op:%s(%s), value:%d is invalid", ATTR_DYNAMIC_TYPE.c_str(),
             switch_n->GetName().c_str(), switch_n->GetType().c_str(), dynamic_type);
      return FAILED;
    }
    if (dynamic_type_ != static_cast<int32_t>(FIXED) && dynamic_type_ != dynamic_type) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%d not same as attr value:%d in node before, "
                         "check invalid",
                         ATTR_DYNAMIC_TYPE.c_str(), switch_n->GetName().c_str(), switch_n->GetType().c_str(),
                         dynamic_type, dynamic_type_);
      GELOGE(FAILED, "[Check][Param] Attr:%s in op:%s(%s), value:%d not same as attr value:%d in node before",
             ATTR_DYNAMIC_TYPE.c_str(), switch_n->GetName().c_str(), switch_n->GetType().c_str(),
             dynamic_type, dynamic_type_);
      return FAILED;
    }
    dynamic_type_ = dynamic_type;
  }
  if (dynamic_type_ == static_cast<int32_t>(FIXED)) {
    REPORT_INNER_ERROR("E19999", "Find Attr:%s in all switcnn node failed", ATTR_DYNAMIC_TYPE.c_str());
    GELOGE(FAILED, "[Check][Param] Find Attr:%s in all switcnn node failed", ATTR_DYNAMIC_TYPE.c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Get user designate shape order. eg{"data","label","mask"}
/// @return Status
///
Status MultiBatchPass::GetUserDesignateShape() {
  data_name_order_.clear();
  bool first_check = true;
  for (const auto &switch_n : switch_n_nodes_) {
    std::vector<std::string> cur_data_name_order;
    if (!AttrUtils::GetListStr(switch_n->GetOpDesc(), ATTR_USER_DESIGNEATE_SHAPE_ORDER, cur_data_name_order)) {
      REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
                        switch_n->GetName().c_str(), switch_n->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
             switch_n->GetName().c_str(), switch_n->GetType().c_str());
      return FAILED;
    }
    if (first_check) {
      data_name_order_ = cur_data_name_order;
      first_check = false;
    } else {
      if (data_name_order_ != cur_data_name_order) {
        REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%s not same as attr value:%s in node before, "
                           "check invalid", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
                           switch_n->GetName().c_str(), switch_n->GetType().c_str(),
                           formats::JoinToString(cur_data_name_order).c_str(),
                           formats::JoinToString(data_name_order_).c_str());
        GELOGE(FAILED, "[Check][Param] Attr:%s in op:%s(%s), value:%s not same as attr value:%s in node before.",
               ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(), switch_n->GetName().c_str(), switch_n->GetType().c_str(),
               formats::JoinToString(cur_data_name_order).c_str(), formats::JoinToString(data_name_order_).c_str());
        return FAILED;
      }
    }
  }
  if (data_name_order_.empty()) {
    REPORT_INNER_ERROR("E19999", "Find Attr:%s in all switcnn node failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str());
    GELOGE(FAILED, "[Check][Param] Find Attr:%s in all switcnn node failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Check SwitchN nodes
/// @param [out] batch_shape
/// @param [out] combined_batch
/// @return bool
///
bool MultiBatchPass::CheckSwitchN(std::vector<std::vector<int64_t>> &batch_shape,
                                  std::vector<std::vector<int64_t>> &combined_batch) {
  // Check if output_num of different SwitchN is same
  uint32_t batch_num = 0;
  for (const NodePtr &node : switch_n_nodes_) {
    uint32_t tmp_num = node->GetAllOutDataAnchorsSize();
    if (batch_num == 0) {
      batch_num = tmp_num;
    } else if (batch_num != tmp_num) {
      REPORT_INNER_ERROR("E19999", "Ouput size num:%u of node:%s(%s) not same as output size num:%d of node before, "
                         "check invalid", tmp_num, node->GetName().c_str(), node->GetType().c_str(), batch_num);
      GELOGE(FAILED, "[Check][Param] Ouput size num:%u of node:%s(%s) not same as output size num:%d of node before",
             tmp_num, node->GetName().c_str(), node->GetType().c_str(), batch_num);
      return false;
    }
  }

  if (!GetBatchInfo(batch_num, batch_shape, combined_batch)) {
    GELOGE(FAILED, "[Get][BatchInfo] failed, batch_num:%u.", batch_num);
    return false;
  }

  if (batch_shape.empty()) {
    REPORT_INNER_ERROR("E19999", "batch_shape size is empty after GetBatchInfo, check invalid");
    GELOGE(FAILED, "[Check][Param] batch_shape is empty after GetBatchInfo.");
    return false;
  }
  if (combined_batch.empty()) {
    REPORT_INNER_ERROR("E19999", "combined_batch size is empty after GetBatchInfo, check invalid");
    GELOGE(FAILED, "[Check][Param] combined_batch is empty after GetBatchInfo.");
    return false;
  }
  size_t dim_num = batch_shape[0].size();
  size_t combined_dim_num = combined_batch[0].size();
  for (uint32_t i = 1; i < batch_num; i++) {
    size_t tmp_dim_num = batch_shape[i].size();
    if (dim_num != tmp_dim_num) {
      REPORT_INNER_ERROR("E19999", "Dim num of batch_shape not equal, batch_0:%zu, batch_%u:%zu, check invalid",
                         dim_num, i, tmp_dim_num);
      GELOGE(FAILED, "[Check][Param] Dim num of batch_shape not equal, batch_0:%zu, batch_%u:%zu.",
             dim_num, i, tmp_dim_num);
      return false;
    }
    size_t tmp_combined_dim_num = combined_batch[i].size();
    if (combined_dim_num != tmp_combined_dim_num) {
      REPORT_INNER_ERROR("E19999", "Dim num of combined_batch not equal, batch_0:%zu, batch_%u:%zu, check invalid",
                         combined_dim_num, i, tmp_combined_dim_num);
      GELOGE(FAILED, "[Check][Param] Dim num of combined_batch not equal, batch_0:%zu, batch_%u:%zu.",
             combined_dim_num, i, tmp_combined_dim_num);
      return false;
    }
  }

  return true;
}

///
/// @brief Check SwitchN nodes
/// @param [in] batch_num
/// @param [out] batch_shape
/// @param [out] combined_batch
/// @return bool
///
bool MultiBatchPass::GetBatchInfo(uint32_t batch_num, std::vector<std::vector<int64_t>> &batch_shape,
                                  std::vector<std::vector<int64_t>> &combined_batch) {
  // Check if output_shape of different SwitchN is same
  std::vector<std::vector<int64_t>> idx_batch_shape;
  std::vector<std::vector<int64_t>> idx_combined_batch;
  for (uint32_t i = 0; i < batch_num; i++) {
    idx_batch_shape.clear();
    idx_combined_batch.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      OpDescPtr op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
        GELOGE(FAILED, "[Get][OpDesc] failed, OpDesc in node is nullptr.");
        return false;
      }
      std::vector<int64_t> output_dims;
      if (!AttrUtils::GetListInt(op_desc->GetOutputDesc(i), ATTR_NAME_SWITCHN_PRED_VALUE, output_dims)) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s from output:%u tensor of op:%s(%s) failed",
                          ATTR_NAME_SWITCHN_PRED_VALUE.c_str(), i,
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][Attr] %s from output:%u tensor of op:%s(%s) failed",
               ATTR_NAME_SWITCHN_PRED_VALUE.c_str(), i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return false;
      }
      idx_batch_shape.emplace_back(output_dims);
      output_dims.clear();
      if (!AttrUtils::GetListInt(op_desc->GetOutputDesc(i), ATTR_NAME_COMBINED_DYNAMIC_DIMS, output_dims)) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s from output:%u tensor of op:%s(%s) failed",
                          ATTR_NAME_COMBINED_DYNAMIC_DIMS.c_str(), i,
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][Attr] %s from output:%u tensor of op:%s(%s) failed",
               ATTR_NAME_COMBINED_DYNAMIC_DIMS.c_str(), i, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return false;
      }
      idx_combined_batch.emplace_back(output_dims);
    }
    if (!CheckDims(idx_batch_shape)) {
      REPORT_INNER_ERROR("E19999", "Attr:%s of all output:%u tensor in switcnn node not equal, or not exist, "
                         "check invalid", ATTR_NAME_SWITCHN_PRED_VALUE.c_str(), i);
      GELOGE(FAILED, "[Check][Dims] failed, Attr:%s of all output:%u tensor in switcnn node not equal, or not exist.",
             ATTR_NAME_SWITCHN_PRED_VALUE.c_str(), i);
      return false;
    }

    batch_shape.emplace_back(idx_batch_shape[0]);
    combined_batch.emplace_back(idx_combined_batch[0]);
  }
  return true;
}

///
/// @brief Find outputs of SwitchN nodes
/// @param [in] batch_num
/// @return void
///
Status MultiBatchPass::FindSwitchOutNodes(uint32_t batch_num) {
  std::vector<NodePtr> output_nodes;
  for (uint32_t i = 0; i < batch_num; i++) {
    output_nodes.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      // idx is promised to be valid
      OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(i);
      GE_CHECK_NOTNULL(out_data_anchor);
      for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto out_node = peer_in_anchor->GetOwnerNode();
        if (out_node->GetType() != IDENTITY || !out_node->GetOutDataNodes().empty()) {
          output_nodes.emplace_back(out_node);
          continue;
        }
        bypass_nodes_.emplace_back(out_node);
        if (GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                            node->GetName().c_str(), node->GetType().c_str(), i,
                            out_node->GetName().c_str(), out_node->GetType().c_str(), peer_in_anchor->GetIdx());
          GELOGE(FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                 node->GetName().c_str(), node->GetType().c_str(), i,
                 out_node->GetName().c_str(), out_node->GetType().c_str(), peer_in_anchor->GetIdx());
          return FAILED;
        }
        for (auto &identity_out_node : out_node->GetOutControlNodes()) {
          output_nodes.emplace_back(identity_out_node);
          if (GraphUtils::RemoveEdge(out_node->GetOutControlAnchor(), identity_out_node->GetInControlAnchor()) !=
              GRAPH_SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                              out_node->GetName().c_str(), out_node->GetType().c_str(),
                              identity_out_node->GetName().c_str(), identity_out_node->GetType().c_str());
            GELOGE(FAILED, "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                   out_node->GetName().c_str(), out_node->GetType().c_str(),
                   identity_out_node->GetName().c_str(), identity_out_node->GetType().c_str());
            return FAILED;
          }
        }
      }
    }
    batch_head_nodes_.emplace_back(output_nodes);
  }

  return SUCCESS;
}

///
/// @brief Replace & Combine SwitchN nodes
/// @param [in] graph
/// @param [in] pred_value
/// @param [in] batch_shape
/// @param [in] combined_batch
/// @return Status
///
Status MultiBatchPass::ReplaceSwitchN(const ComputeGraphPtr &graph, const OutDataAnchorPtr &pred_value,
                                      const std::vector<std::vector<int64_t>> &batch_shape,
                                      const std::vector<std::vector<int64_t>> &combined_batch) {
  NodePtr pred_value_node = pred_value->GetOwnerNode();
  // Create SwitchCase node
  const std::string &switch_case_name = pred_value_node->GetName() + "_" + STREAMSWITCHN;
  NodePtr switch_case = CreateSwitchCaseNode(graph, switch_case_name, pred_value, batch_shape, combined_batch);
  if (switch_case == nullptr) {
    GELOGE(FAILED, "[Create][SwitchCaseNode] %s failed.", switch_case_name.c_str());
    return FAILED;
  }

  for (const NodePtr &switch_n_node : switch_n_nodes_) {
    if (BypassSwitchN(switch_n_node, switch_case) != SUCCESS) {
      GELOGE(FAILED, "[Call][BypassSwitchN] for %s failed.", switch_case_name.c_str());
      return FAILED;
    }
  }

  // Add switchCase input edge
  if (GraphUtils::AddEdge(pred_value, switch_case->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                      pred_value_node->GetName().c_str(), pred_value_node->GetType().c_str(), pred_value->GetIdx(),
                      switch_case->GetName().c_str(), switch_case->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
           pred_value_node->GetName().c_str(), pred_value_node->GetType().c_str(), pred_value->GetIdx(),
           switch_case->GetName().c_str(), switch_case->GetType().c_str());
    return FAILED;
  }

  if (AttachLabel(switch_case) != SUCCESS) {
    GELOGE(FAILED, "[Attach][Label] for node:%s(%s) failed.",
           switch_case->GetName().c_str(), switch_case->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

///
/// @brief Check if output_shape of different SwitchN is same
/// @param [in] output_shape
/// @return bool
///
bool MultiBatchPass::CheckDims(const std::vector<std::vector<int64_t>> &output_shape) const {
  if (output_shape.empty()) {
    GELOGE(FAILED, "[Check][Param] output_shape is empty.");
    return false;
  }

  for (auto iter = output_shape.begin() + 1; iter != output_shape.end(); ++iter) {
    if (output_shape[0] != *iter) {
      return false;
    }
  }
  return true;
}

///
/// @brief Create StreamSwitchN node
/// @param [in] graph
/// @param [in] name
/// @param [in] pred_value
/// @param [in] batch_shape
/// @param [in] combined_batch
/// @return ge::NodePtr
///
NodePtr MultiBatchPass::CreateSwitchCaseNode(const ComputeGraphPtr &graph, const std::string &name,
                                             const OutDataAnchorPtr &pred_value,
                                             const std::vector<std::vector<int64_t>> &batch_shape,
                                             const std::vector<std::vector<int64_t>> &combined_batch) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMSWITCHN);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  GELOGI("Create StreamSwitchN op:%s.", name.c_str());
  OpDescPtr pred_desc = pred_value->GetOwnerNode()->GetOpDesc();
  if (pred_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
    GELOGE(FAILED, "[Get][OpDesc] failed, OpDesc in node is nullptr.");
    return nullptr;
  }
  if (op_desc->AddInputDesc(pred_desc->GetOutputDesc(pred_value->GetIdx())) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  NodePtr switch_case_node = graph->AddNode(op_desc);
  if (switch_case_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  uint32_t batch_num = static_cast<uint32_t>(batch_shape.size());
  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_BATCH_NUM.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_BATCH_NUM.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }
  if (!AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type_)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_DYNAMIC_TYPE.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_DYNAMIC_TYPE.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }
  if (!AttrUtils::SetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, data_name_order_)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }
  for (uint32_t i = 0; i < batch_num; i++) {
    const std::string &attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_name, batch_shape[i])) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", attr_name.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", attr_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
    const std::string &attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    if (!AttrUtils::SetListInt(op_desc, attr_combined_batch, combined_batch[i])) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", attr_combined_batch.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", attr_combined_batch.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }

  return switch_case_node;
}

///
/// @brief Bypass SwitchN node
/// @param [in] switch_n_node
/// @param [in] switch_case
/// @return Status
///
Status MultiBatchPass::BypassSwitchN(const NodePtr &switch_n_node, const NodePtr &switch_case) {
  InDataAnchorPtr in_data_anchor = switch_n_node->GetInDataAnchor(SWITCH_DATA_INPUT);
  if (in_data_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index:%u in data anchor of node:%s(%s) is nullptr, check invalid",
                       SWITCH_DATA_INPUT, switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str());
    GELOGE(FAILED, "[Get][InDataAnchor] failed, Index:%u in data anchor of node:%s(%s) is nullptr",
           SWITCH_DATA_INPUT, switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str());
    return FAILED;
  }
  OutDataAnchorPtr peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_data_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index:%u in data anchor of node:%s(%s), its peer ahcnhor is nullptr, check invalid",
                       SWITCH_DATA_INPUT, switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str());
    GELOGE(FAILED, "[Get][PeerOutAnchor] failed, Index:%u in data anchor of node:%s(%s), its peer ahcnhor is nullptr",
           SWITCH_DATA_INPUT, switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str());
    return FAILED;
  }
  NodePtr data_input = peer_data_anchor->GetOwnerNode();

  // Remove SwitchN data input
  if (GraphUtils::RemoveEdge(peer_data_anchor, in_data_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
                      data_input->GetName().c_str(), data_input->GetType().c_str(), peer_data_anchor->GetIdx(),
                      switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str(), SWITCH_DATA_INPUT);
    GELOGE(FAILED, "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%u) failed",
           data_input->GetName().c_str(), data_input->GetType().c_str(), peer_data_anchor->GetIdx(),
           switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str(), SWITCH_DATA_INPUT);
    return FAILED;
  }
  if (GraphUtils::AddEdge(data_input->GetOutControlAnchor(), switch_case->GetInControlAnchor()) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      data_input->GetName().c_str(), data_input->GetType().c_str(),
                      switch_case->GetName().c_str(), switch_case->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           data_input->GetName().c_str(), data_input->GetType().c_str(),
           switch_case->GetName().c_str(), switch_case->GetType().c_str());
    return FAILED;
  }

  // Add SwitchCase control output
  for (const OutDataAnchorPtr &out_data_anchor : switch_n_node->GetAllOutDataAnchors()) {
    for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      NodePtr data_output = peer_in_anchor->GetOwnerNode();
      if ((GraphUtils::RemoveEdge(out_data_anchor, peer_in_anchor) != GRAPH_SUCCESS) ||
          (GraphUtils::AddEdge(peer_data_anchor, peer_in_anchor) != GRAPH_SUCCESS)) {
        REPORT_CALL_ERROR("E19999", "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) or "
                          "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                          switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str(), out_data_anchor->GetIdx(),
                          data_output->GetName().c_str(), data_output->GetType().c_str(), peer_in_anchor->GetIdx(),
                          data_input->GetName().c_str(), data_input->GetType().c_str(), peer_data_anchor->GetIdx(),
                          data_output->GetName().c_str(), data_output->GetType().c_str(), peer_in_anchor->GetIdx());
        GELOGE(FAILED, "[Replace][Edge] failed, Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) or "
               "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
               switch_n_node->GetName().c_str(), switch_n_node->GetType().c_str(), out_data_anchor->GetIdx(),
               data_output->GetName().c_str(), data_output->GetType().c_str(), peer_in_anchor->GetIdx(),
               data_input->GetName().c_str(), data_input->GetType().c_str(), peer_data_anchor->GetIdx(),
               data_output->GetName().c_str(), data_output->GetType().c_str(), peer_in_anchor->GetIdx());
        return FAILED;
      }
      if (GraphUtils::AddEdge(switch_case->GetOutControlAnchor(), data_output->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          switch_case->GetName().c_str(), switch_case->GetType().c_str(),
                          data_output->GetName().c_str(), data_output->GetType().c_str());
        GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               switch_case->GetName().c_str(), switch_case->GetType().c_str(),
               data_output->GetName().c_str(), data_output->GetType().c_str());
        return FAILED;
      }
    }
  }
  GE_CHK_STATUS_RET(MoveCtrlEdges(switch_n_node, switch_case),
                    "[Move][CtrlEdges] from %s to %s failed.", switch_n_node->GetName().c_str(),
                    switch_case->GetName().c_str());

  bypass_nodes_.emplace_back(switch_n_node);
  GELOGI("Bypass SwitchN node %s success.", switch_n_node->GetName().c_str());
  return SUCCESS;
}

///
/// @brief Attach stream_label & batch_label for batch branch
/// @param [in] switch_case_node
/// @return Status
///
Status MultiBatchPass::AttachLabel(const NodePtr &switch_case_node) {
  std::vector<std::string> stream_label_list;
  for (uint32_t i = 0; i < static_cast<uint32_t>(batch_head_nodes_.size()); i++) {
    if (AttachBatchLabel(i) != SUCCESS) {
      GELOGE(FAILED, "[Attach][BatchLabel] failed, batch_idx=%u", i);
      return FAILED;
    }

    const std::string &stream_label = "stream_label_batch_" + std::to_string(i);
    if (AttachStreamLabel(i, stream_label) != SUCCESS) {
      GELOGE(FAILED, "[Attach][StreamLabel] failed, stream_label=%s, batch_idx=%u", stream_label.c_str(), i);
      return FAILED;
    }
    stream_label_list.emplace_back(stream_label);
  }

  return switch_case_node == nullptr ? SUCCESS : SetActiveLabelList(switch_case_node, stream_label_list);
}

///
/// @brief Attach batch_label for batch branch
/// @param [in] batch_idx
/// @return Status
///
Status MultiBatchPass::AttachBatchLabel(uint32_t batch_idx) {
  std::stack<NodePtr> nodes;
  for (const auto &node : batch_head_nodes_[batch_idx]) {
    nodes.push(node);
  }

  const std::string &batch_label = "Batch_" + std::to_string(batch_idx);
  std::unordered_set<NodePtr> handled_nodes;
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();
    if (handled_nodes.count(cur_node) > 0) {
      continue;
    }

    OpDescPtr cur_desc = cur_node->GetOpDesc();
    GE_CHECK_NOTNULL(cur_desc);
    if (cur_desc->HasAttr(ATTR_NAME_BATCH_LABEL)) {
      std::string tmp_label;
      if (!AttrUtils::GetStr(cur_desc, ATTR_NAME_BATCH_LABEL, tmp_label)) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
                          cur_desc->GetName().c_str(), cur_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
               cur_desc->GetName().c_str(), cur_desc->GetType().c_str());
        return FAILED;
      }
      if (tmp_label != batch_label) {
        REPORT_INNER_ERROR("E19999", "Attr:%s from op:%s(%s) value:%s not equal to expect:%s, check invalid",
                           ATTR_NAME_BATCH_LABEL.c_str(), cur_desc->GetName().c_str(), cur_desc->GetType().c_str(),
                           tmp_label.c_str(), batch_label.c_str());
        GELOGE(FAILED, "[Check][Param] Attr:%s from op:%s(%s) value:%s not equal to expect:%s",
               ATTR_NAME_BATCH_LABEL.c_str(), cur_desc->GetName().c_str(), cur_desc->GetType().c_str(),
               tmp_label.c_str(), batch_label.c_str());
        return FAILED;
      }
    }
    GELOGD("Attach batch_label %s to node %s.", batch_label.c_str(), cur_desc->GetName().c_str());
    if (!AttrUtils::SetStr(cur_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
                        cur_desc->GetName().c_str(), cur_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
             cur_desc->GetName().c_str(), cur_desc->GetType().c_str());
      return FAILED;
    }

    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      OpDescPtr op_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const std::string &type = op_desc->GetType();
      if ((type == MERGE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
        continue;
      }
      if (type == NETOUTPUT) {
        REPORT_CALL_ERROR("E19999", "SReach net_output without Merge, cur_node:%s(%s), check invalid",
                          cur_node->GetName().c_str(), cur_node->GetType().c_str());
        GELOGE(FAILED, "[Check][Param] Reach net_output without Merge, cur_node:%s.", cur_node->GetName().c_str());
        return FAILED;
      }
      nodes.push(out_node);
    }
    (void)handled_nodes.insert(cur_node);
  }

  return SUCCESS;
}

///
/// @brief Attach stream_label for batch branch
/// @param [in] batch_idx
/// @param [in] stream_label
/// @return Status
///
Status MultiBatchPass::AttachStreamLabel(uint32_t batch_idx, const std::string &stream_label) {
  std::stack<NodePtr> nodes;
  for (const auto &node : batch_head_nodes_[batch_idx]) {
    nodes.push(node);
  }

  std::unordered_set<NodePtr> handled_nodes;
  while (!nodes.empty()) {
    NodePtr cur_node = nodes.top();
    nodes.pop();

    OpDescPtr cur_desc = cur_node->GetOpDesc();
    GE_CHECK_NOTNULL(cur_desc);
    if ((handled_nodes.count(cur_node) > 0) || (cur_desc->HasAttr(ATTR_NAME_STREAM_LABEL))) {
      continue;
    }

    GELOGD("Attach stream_label %s to node %s.", stream_label.c_str(), cur_desc->GetName().c_str());
    if (SetStreamLabel(cur_node, stream_label) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                        stream_label.c_str(), cur_node->GetName().c_str(), cur_node->GetType().c_str());
      GELOGE(FAILED, "[Set][StreamLabel] %s to op:%s(%s) failed",
             stream_label.c_str(), cur_node->GetName().c_str(), cur_node->GetType().c_str());
      return FAILED;
    }

    for (const auto &out_node : cur_node->GetOutAllNodes()) {
      nodes.push(out_node);
    }

    (void)handled_nodes.insert(cur_node);
  }

  return SUCCESS;
}

///
/// @brief move edges from old_node to new_node
/// @param [in] old_node
/// @param [in] new_node
/// @return Status
///
Status MultiBatchPass::MoveCtrlEdges(const NodePtr &old_node, const NodePtr &new_node) {
  if (old_node == new_node) {
    return SUCCESS;
  }
  for (const NodePtr &in_ctrl_node : old_node->GetInControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(in_ctrl_node->GetOutControlAnchor(), old_node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  in_ctrl_node->GetName().c_str(), old_node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(in_ctrl_node->GetOutControlAnchor(), new_node->GetInControlAnchor()),
                  "[Add][ControlEdge] between %s and %s failed.",
                  in_ctrl_node->GetName().c_str(), new_node->GetName().c_str());
  }

  for (const NodePtr &out_ctrl_node : old_node->GetOutControlNodes()) {
    GE_CHK_STATUS(GraphUtils::RemoveEdge(old_node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()),
                  "[Remove][ControlEdge] between %s and %s failed.",
                  old_node->GetName().c_str(), out_ctrl_node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor()),
                  "[Add][ControlEdge] between %s and %s failed.",
                  new_node->GetName().c_str(), out_ctrl_node->GetName().c_str());
  }
  return SUCCESS;
}

///
/// @brief attach stream_label & batch_label without change structure of graph
/// @param [in] batch_num
/// @return void
///
Status MultiBatchPass::AttachLabelOnly(uint32_t batch_num) {
  std::vector<NodePtr> output_nodes;
  for (uint32_t i = 0; i < batch_num; i++) {
    output_nodes.clear();
    for (const NodePtr &node : switch_n_nodes_) {
      // idx is promised to be valid
      OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(i);
      GE_CHECK_NOTNULL(out_data_anchor);
      for (const InDataAnchorPtr &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        output_nodes.emplace_back(peer_in_anchor->GetOwnerNode());
      }
    }
    batch_head_nodes_.emplace_back(output_nodes);
  }

  return AttachLabel(nullptr);
}
}  // namespace ge
