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

#include "graph/passes/flow_ctrl_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util.h"
#include "common/ge/ge_util.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/passes/pass_utils.h"

namespace ge {
// when namespace change to ge, please delete the using code.
Status FlowCtrlPass::Run(ComputeGraphPtr compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);

  if (!PassUtils::IsNeedTrainIteFlowCtrl(compute_graph)) {
    GELOGI("No need FlowCtrl for graph %u.", compute_graph->GetGraphID());
    return NOT_CHANGED;
  }

  GELOGI("FlowCtrl pass begin.graph is [%s].", compute_graph->GetName().c_str());
  bool graph_change = false;
  // 1. Add FP/BP flow ctrl (big cycle)
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    uint32_t true_stream_id = 0;
    bool is_found = AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, true_stream_id);
    // FP/BP cycle flag is true_stream_id == 0
    if (is_found && (true_stream_id == TRUE_STREAM_ID)) {
      // Add big cycle
      Status ret = AddFpBpIteratorCtrl(compute_graph, node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Add][FpBpIteratorCtrl] failed, node:%s, graph:%s.",
               node->GetName().c_str(), compute_graph->GetName().c_str());
        return ret;
      }
      graph_change = true;
      // only one big cycle, so break.
      break;
    }
  }

  // 2. Add special node flow ctrl. eg, IteratorGetNext. (small cycle)
  //    NOTE: Small cycle share the variables with big cycle.
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    bool need_cycle_flag = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_STREAM_CYCLE_EVENT_FLAG, need_cycle_flag);
    // small cycle flag is need_stream_cycle_event == true
    if (need_cycle_flag) {
      Status ret = AddSpecialNodeIteratorCtrl(compute_graph, node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Add][SpecialNodeIteratorCtrl] failed, node:%s, graph:%s.",
               node->GetName().c_str(), compute_graph->GetName().c_str());
        return ret;
      }
      graph_change = true;
    }
  }

  // add edge operation below depends on memcpy node in itertor loop set single stream,or may cause block
  for (auto &active_node : active_nodes_in_iter_loop_) {
    auto ret = GraphUtils::AddEdge(active_node->GetOutControlAnchor(),
                                   assign_add_node_in_fpbp_loop_->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGW("add control edge between iter_loop_node:%s and fpbp_loop_node:%s fail, may cause block",
             active_node->GetName().c_str(), assign_add_node_in_fpbp_loop_->GetName().c_str());
    }
  }
  GELOGI("FlowCtrl pass end, graph is %s.", graph_change ? "changed" : "not changed");
  return graph_change ? SUCCESS : NOT_CHANGED;
}

bool FlowCtrlPass::CheckMultiDataSet(ComputeGraphPtr &compute_graph) {
  int data_set_num = 0;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    string type;
    bool is_found = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
    if (is_found && type == "IteratorV2") {
      data_set_num++;
    }
  }
  GELOGI("The ComputeGraph contain %d dataSet.", data_set_num);
  return (data_set_num > 1) ? true : false;
}

NodePtr FlowCtrlPass::InsertOp(ComputeGraphPtr &compute_graph, const string &node_type, const string &node_name,
                               const std::vector<GeTensorDesc> &input_list,
                               const std::vector<GeTensorDesc> &output_list) {
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, node_type);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
    return nullptr;
  }

  for (auto &input_desc : input_list) {
    graphStatus graph_status = op_desc->AddInputDesc(input_desc);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed", op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }

  for (auto &output_desc : output_list) {
    graphStatus graph_status = op_desc->AddOutputDesc(output_desc);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][OutputDesc] to op:%s(%s) failed",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }

  GE_IF_BOOL_EXEC(compute_graph == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
                  DOMI_LOGE("[Check][Param] compute_graph is nullptr");
                  return nullptr);
  NodePtr node = compute_graph->AddNode(op_desc);
  if (node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), compute_graph->GetName().c_str());
    return nullptr;
  }

  GELOGI("Insert op success, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
  return node;
}

NodePtr FlowCtrlPass::InsertStreamSwitchOp(ComputeGraphPtr &compute_graph, const string &switch_name,
                                           const NodePtr &loop_cond, const NodePtr &iter_per_loop) {
  GE_IF_BOOL_EXEC(loop_cond == nullptr || loop_cond->GetOpDesc() == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param loop_cond or its op_desc is nullptr, check invalid");
                  GELOGE(FAILED, "[Check][Param] Param loop_cond or its op_desc is nullptr");
                  return nullptr);
  GE_IF_BOOL_EXEC(iter_per_loop == nullptr || iter_per_loop->GetOpDesc() == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param iter_per_loop or its op_desc is nullptr, check invalid");
                  GELOGE(FAILED, "[Check][Param] Param iter_per_loop or its op_desc is nullptr");
                  return nullptr);
  std::vector<GeTensorDesc> input_desc_list = {loop_cond->GetOpDesc()->GetOutputDesc(0),
                                               iter_per_loop->GetOpDesc()->GetOutputDesc(0)};
  std::vector<GeTensorDesc> output_desc_list;
  NodePtr stream_switch = InsertOp(compute_graph, STREAMSWITCH, switch_name, input_desc_list, output_desc_list);
  if (stream_switch == nullptr) {
    GELOGE(FAILED, "[Insert][StreamSwitchOp] failed, name:%s.", switch_name.c_str());
    return nullptr;
  }

  // set input 0
  graphStatus add_ret = GraphUtils::AddEdge(loop_cond->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(0));
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                      loop_cond->GetName().c_str(), loop_cond->GetType().c_str(),
                      stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
           loop_cond->GetName().c_str(), loop_cond->GetType().c_str(),
           stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
    return nullptr;
  }

  // set input 1
  add_ret = GraphUtils::AddEdge(iter_per_loop->GetOutDataAnchor(0), stream_switch->GetInDataAnchor(1));
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:1) failed",
                      iter_per_loop->GetName().c_str(), iter_per_loop->GetType().c_str(),
                      stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:1) failed",
           iter_per_loop->GetName().c_str(), iter_per_loop->GetType().c_str(),
           stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
    return nullptr;
  }

  // stream switch op need switch cond by attr.
  GE_IF_BOOL_EXEC(!AttrUtils::SetInt(stream_switch->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND,
                                     static_cast<int64_t>(RT_LESS)),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                                    stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
                  DOMI_LOGE("[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                            stream_switch->GetName().c_str(), stream_switch->GetType().c_str());
                  return nullptr);

  return stream_switch;
}

NodePtr FlowCtrlPass::AddVariableNode(ComputeGraphPtr &compute_graph, const string &name) {
  GE_IF_BOOL_EXEC(compute_graph == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
                  DOMI_LOGE("[Check][Param] compute_graph is nullptr");
                  return nullptr);
  NodePtr exist_node = compute_graph->FindNode(name);
  if (exist_node != nullptr) {
    GELOGD("Node %s already exist, no need add.", name.c_str());
    return exist_node;
  }
  // fetch and set tensor desc
  GeTensorDesc tensor_desc;
  if (ge::VarManager::Instance(compute_graph->GetSessionID()) == nullptr) {
    REPORT_INNER_ERROR("E19999", "Get VarManager by session_id:%lu failed", compute_graph->GetSessionID());
    return nullptr;
  }
  Status ret = ge::VarManager::Instance(compute_graph->GetSessionID())->GetCurVarDesc(name, tensor_desc);
  if (ret != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Get var tensor from VarManager by name:%s failed, session_id:%lu",
                       name.c_str(), compute_graph->GetSessionID());
    GELOGE(FAILED, "[Get][CurVarDesc] failed, name:%s, session_id:%lu", name.c_str(), compute_graph->GetSessionID());
    return nullptr;
  }
  std::vector<GeTensorDesc> input_desc_list;
  std::vector<GeTensorDesc> output_desc_list = {tensor_desc};
  // insert node
  return InsertOp(compute_graph, VARIABLE, name, input_desc_list, output_desc_list);
}

Status FlowCtrlPass::AddGlobalStepVariableNode(ComputeGraphPtr &compute_graph) {
  NodePtr output_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    GELOGD("Node type %s can't be found in graph %u", NETOUTPUT, compute_graph->GetGraphID());
    return SUCCESS;
  }
  // Global step just add to main graph's netoutput node.And the main graph must be known shape
  if ((compute_graph->GetParentGraph() != nullptr) ||
      ((compute_graph->GetParentGraph() == nullptr) && (GraphUtils::IsUnknownShapeGraph(compute_graph)))) {
    GELOGD("Subgraph %s no need global step variable.", compute_graph->GetName().c_str());
    return SUCCESS;
  }

  NodePtr exist_node = compute_graph->FindNode(NODE_NAME_GLOBAL_STEP);
  if (exist_node != nullptr) {
    GELOGD("Node %s already exist, no need add.", NODE_NAME_GLOBAL_STEP.c_str());
    return SUCCESS;
  }
  // set global step tensor desc
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_UINT64);
  std::vector<GeTensorDesc> input_desc_list = {};
  std::vector<GeTensorDesc> output_desc_list = {tensor_desc};
  NodePtr global_step = InsertOp(compute_graph, VARIABLE, NODE_NAME_GLOBAL_STEP,
                                 input_desc_list, output_desc_list);
  if (global_step == nullptr) {
    GELOGE(FAILED, "[Insert][Op] in graph:%s failed, global_step is null.", compute_graph->GetName().c_str());
    return FAILED;
  }

  // add ctrl edges
  graphStatus add_ret = GraphUtils::AddEdge(global_step->GetOutControlAnchor(), output_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      global_step->GetName().c_str(), global_step->GetType().c_str(),
                      output_node->GetName().c_str(), output_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           global_step->GetName().c_str(), global_step->GetType().c_str(),
           output_node->GetName().c_str(), output_node->GetType().c_str());
    return FAILED;
  }
  GELOGD("Add global_step to netoutput edge in graph %u success", compute_graph->GetGraphID());
  return SUCCESS;
}

NodePtr FlowCtrlPass::InsertAssignOp(ge::ComputeGraphPtr &compute_graph, const string &node_type,
                                     const string &node_name, const NodePtr &ref_node, const NodePtr &value_node) {
  GE_IF_BOOL_EXEC(ref_node == nullptr || value_node == nullptr ||
                  ref_node->GetOpDesc() == nullptr || value_node->GetOpDesc() == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param ref_node or value_node or their op_desc is nullptr, "
                                     "check invalid");
                  GELOGE(FAILED, "[Check][Param] Param ref_node or value_node or their op_desc is nullptr");
                  return nullptr);
  GeTensorDesc ref_tensor_desc = ref_node->GetOpDesc()->GetOutputDesc(0);
  GeTensorDesc val_tensor_desc = value_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<GeTensorDesc> input_desc_list = {ref_tensor_desc, val_tensor_desc};
  std::vector<GeTensorDesc> output_desc_list = {ref_tensor_desc};
  NodePtr assign_node = InsertOp(compute_graph, node_type, node_name, input_desc_list, output_desc_list);
  if (assign_node == nullptr) {
    GELOGE(FAILED, "[Insert][node] %s(%s) failed.", node_name.c_str(), node_type.c_str());
    return nullptr;
  }
  // assign node input 0 = ref_node
  graphStatus add_ret = GraphUtils::AddEdge(ref_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(0));
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                      ref_node->GetName().c_str(), ref_node->GetType().c_str(),
                      assign_node->GetName().c_str(), assign_node->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
           ref_node->GetName().c_str(), ref_node->GetType().c_str(),
           assign_node->GetName().c_str(), assign_node->GetType().c_str());
    return nullptr;
  }
  // assign input 1 = value_node
  add_ret = GraphUtils::AddEdge(value_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(1));
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:1) failed",
                      value_node->GetName().c_str(), value_node->GetType().c_str(),
                      assign_node->GetName().c_str(), assign_node->GetType().c_str());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:1) failed",
           value_node->GetName().c_str(), value_node->GetType().c_str(),
           assign_node->GetName().c_str(), assign_node->GetType().c_str());
    return nullptr;
  }
  (void)ge::AttrUtils::SetBool(assign_node->GetOpDesc(), ATTR_NEED_COMPILE, true);

  return assign_node;
}

Status FlowCtrlPass::CreateIterCtrlTrueBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                              const NodePtr &loop_inc_node, NodePtr &switch_node) {
  /*
   *           loopCond
   *                |
   *                v
   * switch --> AssignAdd --> active
   *                ^
   *                |
   *         loopIncrement
   */
  // Insert AssignAdd node
  assign_add_node_in_fpbp_loop_ =
      InsertAssignOp(compute_graph, ASSIGNADD, NODE_NAME_FLOWCTRL_LOOP_ASSIGNADD, loop_cond_node, loop_inc_node);
  if (assign_add_node_in_fpbp_loop_ == nullptr || switch_node == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] assign add node or switch node is null");
    return FAILED;
  }

  string active_name = switch_node->GetName() + "_StreamActive";
  // add attr for stream assign model to break branch.
  auto status = SetStreamLabel(assign_add_node_in_fpbp_loop_, active_name);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                      active_name.c_str(), assign_add_node_in_fpbp_loop_->GetName().c_str(),
                      assign_add_node_in_fpbp_loop_->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           active_name.c_str(), assign_add_node_in_fpbp_loop_->GetName().c_str(),
           assign_add_node_in_fpbp_loop_->GetType().c_str());
    return status;
  }

  // used for stream assign to find true branch
  status = SetActiveLabelList(switch_node, { active_name });
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set active label list:%s to op:%s(%s) failed",
                      active_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(status, "[Set][ActiveLabelList] %s to op:%s(%s) failed",
           active_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return status;
  }

  // 2. Insert active node
  NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
  if (active_node == nullptr) {
    GELOGE(FAILED, "[Insert][StreamActiveNode] %s for IterCtrlTrueStream failed.", active_name.c_str());
    return FAILED;
  }
  status = SetStreamLabel(active_node, active_name);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                      active_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           active_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
    return status;
  }
  GE_IF_BOOL_EXEC(!AttrUtils::SetBool(active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                    ATTR_NAME_IS_LOOP_ACTIVE.c_str(),
                                    active_node->GetName().c_str(), active_node->GetType().c_str());
                  DOMI_LOGE("[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_IS_LOOP_ACTIVE.c_str(),
                            active_node->GetName().c_str(), active_node->GetType().c_str());
                  return FAILED);

  // add ctrl edges
  graphStatus add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(),
                                            assign_add_node_in_fpbp_loop_->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      switch_node->GetName().c_str(), switch_node->GetType().c_str(),
                      assign_add_node_in_fpbp_loop_->GetName().c_str(),
                      assign_add_node_in_fpbp_loop_->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(),
           assign_add_node_in_fpbp_loop_->GetName().c_str(), assign_add_node_in_fpbp_loop_->GetType().c_str());
    return FAILED;
  }

  add_ret = GraphUtils::AddEdge(assign_add_node_in_fpbp_loop_->GetOutControlAnchor(),
                                active_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      assign_add_node_in_fpbp_loop_->GetName().c_str(),
                      assign_add_node_in_fpbp_loop_->GetType().c_str(),
                      active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           assign_add_node_in_fpbp_loop_->GetName().c_str(), assign_add_node_in_fpbp_loop_->GetType().c_str(),
           active_node->GetName().c_str(), active_node->GetType().c_str());
    return FAILED;
  }

  GELOGI("CreateIterCtrlTrueBranch success. StreamActive op:%s.", active_node->GetName().c_str());
  return SUCCESS;
}

Status FlowCtrlPass::CreateIterCtrlFalseBranch(ComputeGraphPtr &compute_graph, const NodePtr &loop_cond_node,
                                               const NodePtr &loop_reset_node, NodePtr &switch_node) {
  /*
   *           loopCond
   *                |
   *                v
   *   switch --> Assign --> active --> ModelExit
   *                ^
   *                |
   *            loopReset
   */
  // Insert Assign node and ctrl edge
  NodePtr assign_node =
      InsertAssignOp(compute_graph, ASSIGN, NODE_NAME_FLOWCTRL_LOOP_ASSIGN, loop_cond_node, loop_reset_node);
  if (assign_node == nullptr || switch_node == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] assign_node or switch node is null.");
    return FAILED;
  }

  auto status = SetStreamLabel(assign_node, switch_node->GetName());
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                      switch_node->GetName().c_str(), assign_node->GetName().c_str(), assign_node->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           switch_node->GetName().c_str(), assign_node->GetName().c_str(), assign_node->GetType().c_str());
    return status;
  }

  graphStatus add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), assign_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      switch_node->GetName().c_str(), switch_node->GetType().c_str(),
                      assign_node->GetName().c_str(), assign_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(),
           assign_node->GetName().c_str(), assign_node->GetType().c_str());
    return FAILED;
  }

  if (CheckMultiDataSet(compute_graph)) {
    GELOGI("Multi dataSae exist, model_exit node is need.");
    // 2. Insert active node and add ctrl edge
    string active_name = switch_node->GetName() + "_StreamExitActive";
    NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
    if (active_node == nullptr) {
      GELOGE(FAILED, "[Insert][StreamActiveNode] %s for IterCtrlTrueStream failed.", active_name.c_str());
      return FAILED;
    }
    status = SetStreamLabel(active_node, switch_node->GetName());
    if (status != ge::SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed", switch_node->GetName().c_str(),
                        active_node->GetName().c_str(), active_node->GetType().c_str());
      GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed", switch_node->GetName().c_str(),
             active_node->GetName().c_str(), active_node->GetType().c_str());
      return status;
    }
    GE_CHK_STATUS_RET(SetSwitchBranchNodeLabel(active_node, switch_node->GetName()),
                      "[Set][SwitchBranchNodeLabel] %s to op:%s(%s) failed", switch_node->GetName().c_str(),
                      active_node->GetName().c_str(), active_node->GetType().c_str());

    string model_exit_name = switch_node->GetName() + "_ModelExit";
    status = SetActiveLabelList(active_node, { model_exit_name });
    if (status != ge::SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Set active label list:%s to op:%s(%s) failed",
                        model_exit_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
      GELOGE(status, "[Set][ActiveLabelList] %s to op:%s(%s) failed",
             model_exit_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
      return status;
    }

    add_ret = GraphUtils::AddEdge(assign_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
    if (add_ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        assign_node->GetName().c_str(), assign_node->GetType().c_str(),
                        active_node->GetName().c_str(), active_node->GetType().c_str());
      GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             assign_node->GetName().c_str(), assign_node->GetType().c_str(),
             active_node->GetName().c_str(), active_node->GetType().c_str());
      return FAILED;
    }

    // 3. Insert model exit node and add ctrl edge
    NodePtr model_exit_node = InsertOp(compute_graph, MODELEXIT, model_exit_name, {}, {});
    if (model_exit_node == nullptr) {
      GELOGE(FAILED, "[Insert][ModelExitNode] %s for IterCtrlTrueStream failed.", model_exit_name.c_str());
      return FAILED;
    }
    status = SetStreamLabel(model_exit_node, model_exit_name);
    if (status != ge::SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Set stream_label:%s to op:%s(%s) failed",
                        model_exit_name.c_str(), model_exit_node->GetName().c_str(),
                        model_exit_node->GetType().c_str());
      GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed", model_exit_name.c_str(),
             model_exit_node->GetName().c_str(), model_exit_node->GetType().c_str());
      return status;
    }

    add_ret = GraphUtils::AddEdge(active_node->GetOutControlAnchor(), model_exit_node->GetInControlAnchor());
    if (add_ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        active_node->GetName().c_str(), assign_node->GetType().c_str(),
                        model_exit_node->GetName().c_str(), model_exit_node->GetType().c_str());
      GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             active_node->GetName().c_str(), assign_node->GetType().c_str(),
             model_exit_node->GetName().c_str(), model_exit_node->GetType().c_str());
      return FAILED;
    }
  }

  GELOGI("CreateIterCtrlFalseBranch success.");
  return SUCCESS;
}

Status FlowCtrlPass::AddFpBpIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &pre_node) {
  GE_IF_BOOL_EXEC(pre_node == nullptr, DOMI_LOGE("[Check][Param] pre_node is nullptr."); return FAILED);
  string pre_node_name = pre_node->GetName();
  GELOGI("Add FpBp Iterator ctrl, pre node:%s.", pre_node_name.c_str());
  // 1. Get or add variables
  NodePtr loop_cond_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_COND);
  if (loop_cond_node == nullptr) {
    GELOGE(FAILED, "[Add][Variable] %s failed.", NODE_NAME_FLOWCTRL_LOOP_COND.c_str());
    return FAILED;
  }
  NodePtr loop_inc_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_INCREMENT);
  if (loop_inc_node == nullptr) {
    GELOGE(FAILED, "[Add][Variable] %s failed.", NODE_NAME_FLOWCTRL_LOOP_INCREMENT.c_str());
    return FAILED;
  }
  NodePtr loop_reset_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_RESETVALUE);
  if (loop_reset_node == nullptr) {
    GELOGE(FAILED, "[Add][Variable] %s failed.", NODE_NAME_FLOWCTRL_LOOP_RESETVALUE.c_str());
    return FAILED;
  }
  NodePtr iter_per_loop_node = AddVariableNode(compute_graph, NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  if (iter_per_loop_node == nullptr) {
    GELOGE(FAILED, "[Add][Variable] %s failed.", NODE_NAME_FLOWCTRL_LOOP_PER_ITER.c_str());
    return FAILED;
  }

  // 2. Add StreamSwitch
  string switch_name = pre_node_name + "_" + NODE_NAME_STREAM_SWITCH;
  NodePtr switch_node = InsertStreamSwitchOp(compute_graph, switch_name, loop_cond_node, iter_per_loop_node);
  if (switch_node == nullptr) {
    GELOGE(FAILED, "[Insert][StreamSwitchOp] %s failed.", switch_name.c_str());
    return FAILED;
  }
  auto status = SetStreamLabel(switch_node, switch_name);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                      switch_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           switch_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return status;
  }

  graphStatus add_ret = GraphUtils::AddEdge(pre_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                      switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           pre_node->GetName().c_str(), pre_node->GetType().c_str(),
           switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return FAILED;
  }

  // 3. Create switch false branch: return results and reset the loopCond
  Status ret = CreateIterCtrlFalseBranch(compute_graph, loop_cond_node, loop_reset_node, switch_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Create][IterCtrlFalseBranch] fail, pre node:%s.", pre_node_name.c_str());
    return ret;
  }

  // 4. Create switch true branch:
  // active train streams and increase the loopCond
  ret = CreateIterCtrlTrueBranch(compute_graph, loop_cond_node, loop_inc_node, switch_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Create][IterCtrlTrueBranch] fail, pre node:%s.", pre_node_name.c_str());
    return ret;
  }
  return SUCCESS;
}

Status FlowCtrlPass::AddSpecialNodeIteratorCtrl(ComputeGraphPtr &compute_graph, NodePtr &loop_after_node) {
  /*
   * before add:
   *    iterator
   *       |
   *       v
   *   MemcpyAsync
   *
   * after add:
   *    iterator ----------┐
   *       |               ┆c
   *       v        c      v      c
   *   MemcpyAsync-----> switch -----> active
   *                       ^
   *                     /   \
   *          itersPerLoop  loopCond
   */
  GE_IF_BOOL_EXEC(loop_after_node == nullptr || compute_graph == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param loop_after_node or compute_graph is nullptr, check invalid");
                  DOMI_LOGE("[Check][Param] loop after node or compute graph is null.");
                  return FAILED);
  InDataAnchorPtr in_anchor = loop_after_node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param loop_after_node:%s(%s) no in data node, check invalid",
                       loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Param loop_after_node:%s(%s) no in data node.",
           loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str());
    return FAILED;
  }
  NodePtr loop_pre_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();

  // 1. Get variables
  NodePtr loop_cond_node = compute_graph->FindNode(NODE_NAME_FLOWCTRL_LOOP_COND);
  if (loop_cond_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s not found in graph:%s, check invalid",
                       NODE_NAME_FLOWCTRL_LOOP_COND.c_str(), compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] Node:%s not found in graph:%s.",
           NODE_NAME_FLOWCTRL_LOOP_COND.c_str(), compute_graph->GetName().c_str());
    return FAILED;
  }
  NodePtr iter_per_loop_node = compute_graph->FindNode(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
  if (iter_per_loop_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Node:%s not found in graph:%s, check invalid",
                       NODE_NAME_FLOWCTRL_LOOP_PER_ITER.c_str(), compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] Node:%s not found in graph:%s.",
           NODE_NAME_FLOWCTRL_LOOP_PER_ITER.c_str(), compute_graph->GetName().c_str());
    return FAILED;
  }

  // 2. Add StreamSwitch and edges to switch_node.
  GE_IF_BOOL_EXEC(loop_pre_node == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param loop_after_node:%s(%s) no in data node, check invalid",
                                     loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str());
                  DOMI_LOGE("[Check][Param] Param loop_after_node:%s(%s) no in data node",
                            loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str());
                  return FAILED);
  string switch_name = loop_pre_node->GetName() + "_" + NODE_NAME_STREAM_SWITCH;
  NodePtr switch_node = InsertStreamSwitchOp(compute_graph, switch_name, loop_cond_node, iter_per_loop_node);
  if (switch_node == nullptr) {
    GELOGE(FAILED, "[Insert][StreamSwitchOp] %s failed.", switch_name.c_str());
    return FAILED;
  }

  auto status = SetStreamLabel(switch_node, switch_name);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                      switch_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           switch_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return status;
  }

  graphStatus add_ret = GraphUtils::AddEdge(loop_pre_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      loop_pre_node->GetName().c_str(), loop_pre_node->GetType().c_str(),
                      switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           loop_pre_node->GetName().c_str(), loop_pre_node->GetType().c_str(),
           switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return FAILED;
  }
  add_ret = GraphUtils::AddEdge(loop_after_node->GetOutControlAnchor(), switch_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str(),
                      switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           loop_after_node->GetName().c_str(), loop_after_node->GetType().c_str(),
           switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return FAILED;
  }

  // 3. Create switch true branch: only active
  string active_name = switch_name + "_StreamActive";
  NodePtr active_node = InsertOp(compute_graph, STREAMACTIVE, active_name, {}, {});
  if (active_node == nullptr) {
    GELOGE(FAILED, "[Insert][StreamActiveNode] %s for SpecialNodeIteratorCtrl failed.", active_name.c_str());
    return FAILED;
  }

  status = SetStreamLabel(active_node, active_name);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                      active_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
           active_name.c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
    return status;
  }

  GE_IF_BOOL_EXEC(!AttrUtils::SetBool(active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true),
                  REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_IS_LOOP_ACTIVE.c_str(),
                                    active_node->GetName().c_str(), active_node->GetType().c_str());
                  DOMI_LOGE("[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_IS_LOOP_ACTIVE.c_str(),
                            active_node->GetName().c_str(), active_node->GetType().c_str());
                  return FAILED);

  add_ret = GraphUtils::AddEdge(switch_node->GetOutControlAnchor(), active_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      switch_node->GetName().c_str(), switch_node->GetType().c_str(),
                      active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           switch_node->GetName().c_str(), switch_node->GetType().c_str(),
           active_node->GetName().c_str(), active_node->GetType().c_str());
    return FAILED;
  }

  // used for stream assign to find true branch
  status = SetActiveLabelList(switch_node, { active_name });
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set active label list:%s to op:%s(%s) failed",
                      active_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(status, "[Set][ActiveLabelList] %s to op:%s(%s) failed",
           active_name.c_str(), switch_node->GetName().c_str(), switch_node->GetType().c_str());
    return status;
  }
  // used for stream assign to find active stream
  status = SetActiveLabelList(active_node, { loop_pre_node->GetName() });
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set active label list:%s to op:%s(%s) failed", loop_pre_node->GetName().c_str(),
                      active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(status, "[Set][ActiveLabelList] %s to op:%s(%s) failed",
           loop_pre_node->GetName().c_str(), active_node->GetName().c_str(), active_node->GetType().c_str());
    return status;
  }
  active_nodes_in_iter_loop_.push_back(active_node);
  return SUCCESS;
}
}  // namespace ge
