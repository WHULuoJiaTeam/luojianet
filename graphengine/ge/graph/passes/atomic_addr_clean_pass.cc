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

#include "graph/passes/atomic_addr_clean_pass.h"

#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "common/ge_call_wrapper.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "init/gelib.h"

namespace ge {
Status AtomicAddrCleanPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGD("AtomicAddrCleanPass begin.");
  // 1.Recoginze atomic and loop mark
  vector<NodePtr> atomic_node_vec;
  for (NodePtr &node : graph->GetDirectNode()) {
    if (IsAtomicOp(node)) {
      atomic_node_vec.push_back(node);
    }
    if (!is_loop_graph_ && node->GetType() == LOOPCOND) {
      // there is loop in this graph
      GELOGD("There is no loop node. It will insert clean node follow atomic node.");
      is_loop_graph_ = true;
    }
  }
  if (atomic_node_vec.empty()) {
    GELOGD("There is no atomic node. Ignore atomicAddrClean pass.");
    return SUCCESS;
  }

  bool is_unknown_graph = graph->GetGraphUnknownFlag();
  if (is_unknown_graph) {
    GELOGD("Graph[%s] is unknown graph. It will call fe interface to compile op.", graph->GetName().c_str());
    GE_CHK_STATUS_RET(CompileUnknownGraphOp(atomic_node_vec));
    return SUCCESS;
  }

  // 2.Insert clean node and link to atomic node
  Status ret;
  if (is_loop_graph_) {
    ret = HandleLoopGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  } else {
    ret = HandleNormalGraph(graph, atomic_node_vec);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  GELOGD("AtomicAddrCleanPass end.");
  return SUCCESS;
}

// just hccl may mark atomic from ops kernel now, and hccl's atomic if for all input
bool AtomicAddrCleanPass::CheckAtomicFromOpsKernel(const NodePtr &node) {
  // 1.Check if isAtomic attrs exist for HCOM
  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized, atomic from ops kernel judge false, node_name: %s", node->GetName().c_str());
    return false;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  vector<OpInfo> op_info_vec = ops_kernel_manager.GetOpsKernelInfo(node->GetType());
  for (const auto &op_info : op_info_vec) {
    if (op_info.isAtomic) {
      // check peer input is DATA
      for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
        if (in_data_anchor->GetPeerOutAnchor() != nullptr &&
            in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() != nullptr) {
          auto peer_in_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
          if (peer_in_node->GetType() == DATA) {
            GELOGI("Recognized atomic op %s from %s engine and input is DATA.", node->GetName().c_str(),
                   op_info.engine.c_str());
            return false;
          }
        }
      }
      GELOGI("Recognized atomic op %s from %s engine.", node->GetName().c_str(), op_info.engine.c_str());
      hcom_node_vec_.push_back(node);
      return true;
    }
  }
  return false;
}

bool AtomicAddrCleanPass::IsOutputIndexPeerInputAtomic(const NodePtr &node, int64_t output_index) {
  auto out_data_anchor = node->GetAllOutDataAnchors().at(output_index);
  if (out_data_anchor == nullptr) {
    return false;
  }

  for (auto input_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    auto output_node = input_anchor->GetOwnerNode();
    // just hccl may mark atomic from ops kernel now, and hccl's atomic if for all input
    // hccl's attr ATOMIC_ATTR_INPUT_INDEX mark on CalcOpRunningParam, can't be get here
    if (CheckAtomicFromOpsKernel(output_node)) {
      return true;
    }
  }
  return false;
}

bool AtomicAddrCleanPass::CheckSkipInsertInLoopGraph(const NodePtr &node) {
  OpDescPtr op_desc = node->GetOpDesc();
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  bool has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  bool has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  if (!has_atomic_input && has_atomic_output && atomic_workspace_index_size.empty()) {
    std::vector<int64_t> atomic_output_index;
    (void) ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    bool is_all_output_peer_also_atomic = true;
    for (const auto &output_index : atomic_output_index) {
      if (!IsOutputIndexPeerInputAtomic(node, output_index)) {
        is_all_output_peer_also_atomic = false;
        break;
      }
    }
    if (is_all_output_peer_also_atomic) {
      GELOGI("all out peer node input atomic, skip this out atomic process, node name: %s", node->GetName().c_str());
      return true;
    }
  }
  return false;
}

Status AtomicAddrCleanPass::HandleLoopGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec) {
  // Loop graph , insert clean node follow atomic node
  int index = 0;
  for (const auto &node : atomic_node_vec) {
    if (CheckSkipInsertInLoopGraph(node)) {
      continue;
    }

    // Insert atomic clean op
    NodePtr clean_addr_node = InsertAtomicAddrCleanNode(graph);
    if (clean_addr_node == nullptr) {
      GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] to graph:%s failed. Ignore atomicAddrClean pass.",
             graph->GetName().c_str());
      return FAILED;
    }

    GE_CHECK_NOTNULL(clean_addr_node->GetOpDesc());
    string node_name = clean_addr_node->GetOpDesc()->GetName();
    std::ostringstream oss;
    oss << node_name << index;
    node_name = oss.str();
    clean_addr_node->GetOpDesc()->SetName(node_name);  // [Cascade Pointer]
    GELOGD("Inserted atomic clean node name is %s", node_name.c_str());

    auto ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret,
             "[Call][LinkToAtomicNode] Link control anchor failed from atomic node:%s to atomic_addr_clean:%s node.",
             node->GetName().c_str(), clean_addr_node->GetName().c_str());
      return ret;
    }
    index++;
  }
  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleNormalGraph(ComputeGraphPtr &graph, const vector<NodePtr> &atomic_node_vec) {
  GELOGD("Not loop graph and unknown graph. It will insert atomic clean nodes.");

  vector<NodePtr> common_atomic_nodes;
  vector<NodePtr> dispersed_atomic_nodes;
  auto ret = HandleDispersedAtomicNodes(graph, atomic_node_vec, common_atomic_nodes, dispersed_atomic_nodes);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][HandleDispersedAtomicNodes] failed, graph name is %s.", graph->GetName().c_str());
    return ret;
  }

  if (common_atomic_nodes.empty()) {
    GELOGI("common_atomic_nodes is empty");
    return SUCCESS;
  }

  // not loop graph , insert only one clean node in graph
  NodePtr clean_addr_node = InsertAtomicAddrCleanNode(graph);
  if (clean_addr_node == nullptr) {
    GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] in graph:%s failed. Ignore atomicAddrClean pass.",
           graph->GetName().c_str());
    return FAILED;
  }
  for (const auto &node : common_atomic_nodes) {
    ret = LinkToAtomicNode(node, clean_addr_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Link][ControlAnchor] failed from atomic node:%s to atomic_addr_clean node:%s.",
             node->GetName().c_str(), clean_addr_node->GetName().c_str());
      return ret;
    }
  }

  // for HCOM atomic node, add one more control link to peer-in node
  for (auto &node : hcom_node_vec_) {
    for (auto &in_anchor : node->GetAllInDataAnchors()) {
      GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
      NodePtr peer_in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      ret = LinkToAtomicNode(peer_in_node, clean_addr_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Link][ControlAnchor] from node:%s to node:%s failed",
               peer_in_node->GetName().c_str(), clean_addr_node->GetName().c_str());
        return ret;
      }
    }
  }
  return LinkToPotentialPrecedenceNode(graph, clean_addr_node, dispersed_atomic_nodes);
}

// Add control edges from atomic clean node to all potential precedence nodes which may execute before atomic clean
// node. We hope that atomic clean node can execute with the highest priority in the entire graph. Because of stream
// concurrency mechanism, only placing it at the head can not ensure that priority. Therefore, we need to add control
// edges from atomic clean node to the nodes that may be the first node on each stream. Generally, the first nodes on
// each stream are successors of Data/Variable, and Data/Variable won't generate task or execute, so we link to the
// successors of Data/Variable.
Status AtomicAddrCleanPass::LinkToPotentialPrecedenceNode(ComputeGraphPtr &graph, NodePtr &atomic_clean_node,
                                                          const vector<NodePtr> &dispersed_atomic_nodes) {
  GELOGD("Start to add control edges from %s to all second-nodes behind first-nodes which have no input.",
         atomic_clean_node->GetName().c_str());
  auto out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctrl_anchor);

  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    bool need_handle = (node->GetType() == DATA || node->GetType() == VARIABLE) && node->GetInAllNodes().empty();
    if (!need_handle) {
      continue;
    }
    auto second_nodes = node->GetOutAllNodes();
    for (const auto &second_node : second_nodes) {
      GE_CHECK_NOTNULL(second_node);
      if ((std::find(dispersed_atomic_nodes.begin(), dispersed_atomic_nodes.end(), second_node) !=
          dispersed_atomic_nodes.end()) || (second_node->GetType() == NETOUTPUT)) {
        continue;
      }
      auto in_ctrl_anchor = second_node->GetInControlAnchor();
      GE_CHECK_NOTNULL(in_ctrl_anchor);
      if (!out_ctrl_anchor->IsLinkedWith(in_ctrl_anchor)) {
        GE_CHK_STATUS_RET(out_ctrl_anchor->LinkTo(in_ctrl_anchor));
        GELOGD("Add control edge from %s to %s.", atomic_clean_node->GetName().c_str(), second_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status AtomicAddrCleanPass::HandleDispersedAtomicNodes(ComputeGraphPtr &graph,
                                                       const std::vector<NodePtr> &atomic_node_vec,
                                                       std::vector<NodePtr> &common_atomic_nodes,
                                                       vector<NodePtr> &dispersed_atomic_nodes) {
  int index = 0;
  for (const auto &node : atomic_node_vec) {
    vector<int> node_anchors_connect_netoutput;
    // If GetBool fail, attr is_connect_netoutput is an empty vector.
    (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, node_anchors_connect_netoutput);
    if (!node_anchors_connect_netoutput.empty()) {
      NodePtr dispersed_clean_addr_node = InsertAtomicAddrCleanNode(graph);
      if (dispersed_clean_addr_node == nullptr) {
        GELOGE(FAILED, "[Insert][AtomicAddrCleanNode] in graph:%s failed. Ignore atomicAddrClean pass.",
               graph->GetName().c_str());
        return FAILED;
      }

      auto dispersed_node_op_desc = dispersed_clean_addr_node->GetOpDesc();
      GE_CHECK_NOTNULL(dispersed_node_op_desc);
      string node_name = dispersed_node_op_desc->GetName();
      std::ostringstream oss;
      oss << node_name << "_" << index;
      node_name = oss.str();
      dispersed_node_op_desc->SetName(node_name);
      GELOGD("Inserted dispersed atomic clean node [%s] before [%s]", node_name.c_str(), node->GetName().c_str());
      ++index;
      Status ret = LinkToAtomicNode(node, dispersed_clean_addr_node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Link][ControlAnchor] failed from atomic node:%s to atomic_addr_clean node:%s.",
               node->GetName().c_str(), dispersed_clean_addr_node->GetName().c_str());
        return ret;
      }
      dispersed_atomic_nodes.emplace_back(node);
    } else {
      common_atomic_nodes.emplace_back(node);
    }
  }

  return SUCCESS;
}

NodePtr AtomicAddrCleanPass::InsertAtomicAddrCleanNode(ComputeGraphPtr &graph) {
  OpDescPtr op_desc = MakeShared<OpDesc>(NODE_NAME_ATOMIC_ADDR_CLEAN, ATOMICADDRCLEAN);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(INTERNAL_ERROR, "[New][OpDesc] failed.");
    return nullptr;
  }
  string session_graph_id;
  if (!AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Get graph session_graph_id attr failed.");
  }
  if (!session_graph_id.empty()) {
    (void) AttrUtils::SetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  string node_name = op_desc->GetName();
  // Only flush subgraph name
  if (graph->GetParentGraph() != nullptr) {
    node_name = graph->GetName() + "_" + node_name;
  }

  string name = node_name + session_graph_id;
  op_desc->SetName(name);
  GELOGI("Create cleanAddr op:%s.", op_desc->GetName().c_str());
  // To avoid same name between graphs, set session graph id to this node
  NodePtr clean_addr_node = graph->AddNodeFront(op_desc);
  return clean_addr_node;
}

Status AtomicAddrCleanPass::LinkToAtomicNode(const NodePtr &atomic_node, NodePtr &atomic_clean_node) {
  GE_IF_BOOL_EXEC(atomic_node == nullptr || atomic_clean_node == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param atomic_node or atomic_clean_node is nullptr, "
                                     "check invalid");
                  DOMI_LOGE("[Check][Param] param [atomic_node][atomic_clean_node] must not be null.");
                  return PARAM_INVALID);
  InControlAnchorPtr in_ctrl_anchor = atomic_node->GetInControlAnchor();
  OutControlAnchorPtr out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  if (in_ctrl_anchor == nullptr || out_ctrl_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "in_ctrl_anchor of op:%s(%s) or out_ctrl_anchor of op:%s(%s) is nullptr, "
                       "check invalid",
                       atomic_node->GetName().c_str(), atomic_node->GetType().c_str(),
                       atomic_clean_node->GetName().c_str(), atomic_clean_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] in_ctrl_anchor of op:%s(%s) or out_ctrl_anchor of op:%s(%s) is nullptr.",
           atomic_node->GetName().c_str(), atomic_node->GetType().c_str(),
           atomic_clean_node->GetName().c_str(), atomic_clean_node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  graphStatus status = GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
  if (status != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                      out_ctrl_anchor->GetOwnerNode()->GetType().c_str(),
                      in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                      in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           out_ctrl_anchor->GetOwnerNode()->GetName().c_str(), out_ctrl_anchor->GetOwnerNode()->GetType().c_str(),
           in_ctrl_anchor->GetOwnerNode()->GetName().c_str(), in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Graph add cleanAddrNode op out ctrl edge, dst node: %s.", atomic_node->GetName().c_str());
  std::string stream_label;
  if (is_loop_graph_ && AttrUtils::GetStr(atomic_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    if (!AttrUtils::SetStr(atomic_clean_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
      GELOGW("LinkToAtomicNode: SetStr failed");
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool AtomicAddrCleanPass::IsAtomicOp(const NodePtr &node) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(FAILED, "[Check][Param] node is nullptr."); return false);
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  // 1.Check if isAtomic attrs exist for HCOM
  if (CheckAtomicFromOpsKernel(node)) {
    return true;
  }

  // 2.Check atomic attr in node
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  bool has_atomic_input = op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  bool has_atomic_output = op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
  atomic_workspace_index_size = op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  if (!has_atomic_input && !has_atomic_output && atomic_workspace_index_size.empty()) {
    return false;
  }

  graphStatus ret = op_desc->SetAttr(ATOMIC_ATTR_IS_ATOMIC_NODE, GeAttrValue::CreateFrom<GeAttrValue::BOOL>(true));
  if (ret != GRAPH_SUCCESS) {
    GELOGW("set attr ATOMIC_ATTR_IS_ATOMIC_NODE fail.");
  }
  GELOGD("Recognized atomic op %s from FE engine.", op_desc->GetName().c_str());
  return true;
}
///
/// @brief Clear Status, used for subgraph pass
/// @return SUCCESS
///
Status AtomicAddrCleanPass::ClearStatus() {
  hcom_node_vec_.clear();
  return SUCCESS;
}

Status AtomicAddrCleanPass::CompileUnknownGraphOp(const vector<NodePtr> &atomic_node_vec) {
  GE_TIMESTAMP_CALLNUM_START(UnknownGraphCompileOp);
  std::unordered_map<string, vector<ge::NodePtr>> node_vector_map;
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if ((instance == nullptr) || !instance->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "GeLib is not init before, check invalid");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GeLib is not init before.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }

  for (auto &atomic_node: atomic_node_vec) {
    auto op_desc = atomic_node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGW("op desc is nullptr.");
      continue;
    }
    string kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      REPORT_INNER_ERROR("E19999", "Find ops kernel by name of op:%s(%s) failed",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(ge::INTERNAL_ERROR, "[Get][OpKernelLibName] of op:%s(%s) failed.", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return ge::INTERNAL_ERROR;
    }

    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    node_vector_map[kernel_lib_name].emplace_back(atomic_node);
  }

  for (auto &it : node_vector_map) {
    auto &kernel_lib_name = it.first;
    auto &node_vector = it.second;
    OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
    GE_CHECK_NOTNULL(kernel_info);
    GE_TIMESTAMP_RESTART(UnknownGraphCompileOp);
    auto ret = kernel_info->CompileOp(node_vector);
    GELOGI("The atomic node size of compile op of %s is %zu", kernel_lib_name.c_str(), node_vector.size());
    GE_TIMESTAMP_ADD(UnknownGraphCompileOp);
    if (ret != ge::SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Call CompileOp failed, kernel_lib_name:%s, ret:%d",
                        kernel_lib_name.c_str(), ret);
      GELOGE(ret, "[Compile][AtomicOp] failed, kernel lib name is %s", kernel_lib_name.c_str());
      return ret;
    }
  }
  GE_TIMESTAMP_CALLNUM_END(UnknownGraphCompileOp, "AtomicAddrCleanPass::CompileUnknownGraphOp");
  return SUCCESS;
}
}  // namespace ge
