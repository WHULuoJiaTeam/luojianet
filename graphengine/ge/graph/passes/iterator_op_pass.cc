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

#include "graph/passes/iterator_op_pass.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "common/omg_util.h"
#include "external/graph/graph.h"
#include "graph/node.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"
#include "runtime/mem.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"

namespace ge {
const char *const kGetNext = "GetNext";
const int kMaxIterationsPerLoop = INT32_MAX - 1;

Status IteratorOpPass::Run(ge::ComputeGraphPtr graph) {
  GELOGD("GetNextOpPass begin");
  GE_CHECK_NOTNULL(graph);
  if (!PassUtils::IsNeedTrainIteFlowCtrl(graph)) {
    return SUCCESS;
  }
  std::string type;
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(GetOriginalType(node, type));
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const string op_type = op_desc->GetType();
    if (type == "IteratorV2" || type == "Iterator" || op_type == kGetNext) {
      ge::NodePtr memcpy_async_node = InsertMemcpyAsyncNode(node, graph);
      GE_CHECK_NOTNULL(memcpy_async_node);
      auto status = SetCycleEvent(memcpy_async_node);
      if (status != ge::SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set cycle event to op:%s(%s) failed",
                          memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str());
        GELOGE(status, "[Set][CycleEvent] to op:%s(%s) failed",
               memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str());
        return status;
      }

      status = SetStreamLabel(memcpy_async_node, memcpy_async_node->GetName());
      if (status != ge::SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                          memcpy_async_node->GetName().c_str(), memcpy_async_node->GetName().c_str(),
                          memcpy_async_node->GetType().c_str());
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
               memcpy_async_node->GetName().c_str(), memcpy_async_node->GetName().c_str(),
               memcpy_async_node->GetType().c_str());
        return status;
      }

      status = SetStreamLabel(node, node->GetName());
      if (status != ge::SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                          node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
               node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return status;
      }

      GELOGI("Set independent loop for iterator node success");

      int64_t loop_per_iter = 0;
      ge::GeTensorDesc ge_tensor_desc;
      status = VarManager::Instance(graph->GetSessionID())->GetCurVarDesc(NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
                                                                                 ge_tensor_desc);
      GE_IF_BOOL_EXEC(status != SUCCESS, GELOGW("Fail to Get var_desc of NODE_NAME_FLOWCTRL_LOOP_PER_ITER failed.");
                      continue);
      Status ret;
      ret = SetRtContext(graph->GetSessionID(), graph->GetGraphID(), rtContext_t(), RT_CTX_NORMAL_MODE);

      // EOS will not be considered if ret is not SUCCESS.
      GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGW("Set rt context RT_CTX_NORMAL_MODE failed."); continue);

      status = GetVariableValue(graph->GetSessionID(), ge_tensor_desc, NODE_NAME_FLOWCTRL_LOOP_PER_ITER,
                                &loop_per_iter);
      ret = SetRtContext(graph->GetSessionID(), graph->GetGraphID(), rtContext_t(), RT_CTX_GEN_MODE);

      // The following process will be affected if ret is not SUCCESS.
      GE_IF_BOOL_EXEC(ret != SUCCESS,
                      GELOGE(ret, "[Set][RtContext] RT_CTX_GEN_MODE failed, session_id:%lu, graph_id:%u.",
                             graph->GetSessionID(), graph->GetGraphID());
                      return ret);

      GE_IF_BOOL_EXEC(status != SUCCESS, GELOGW("Get variable value of NODE_NAME_FLOWCTRL_LOOP_PER_ITER failed.");
                      continue);
      GELOGI("The value of NODE_NAME_FLOWCTRL_LOOP_PER_ITER is %ld", loop_per_iter);

      if (loop_per_iter == kMaxIterationsPerLoop) {
        ge::NodePtr end_of_sequence_node = InsertEndOfSequenceNode(node, memcpy_async_node, graph);
        GE_CHECK_NOTNULL(end_of_sequence_node);
        status = SetStreamLabel(end_of_sequence_node, end_of_sequence_node->GetName());
        if (status != ge::SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Set stream label:%s to op:%s(%s) failed",
                            end_of_sequence_node->GetName().c_str(), end_of_sequence_node->GetName().c_str(),
                            end_of_sequence_node->GetType().c_str());
          GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
                 end_of_sequence_node->GetName().c_str(), end_of_sequence_node->GetName().c_str(),
                 end_of_sequence_node->GetType().c_str());
          return status;
        }
        GELOGI("Insert EndOfSequence node success.");
      }
    }
    GELOGI("GetNextOpPass end");
  }
  GELOGD("GetNextOpPass end");
  return SUCCESS;
}

Status IteratorOpPass::GetVariableValue(uint64_t session_id, const ge::GeTensorDesc &tensor_desc,
                                        const std::string &var_name, void *dest) {
  // base_addr
  uint8_t *var_mem_base = VarManager::Instance(session_id)->GetVarMemoryBase(RT_MEMORY_HBM);
  GE_CHECK_NOTNULL(var_mem_base);
  // offset + logic_base
  uint8_t *dev_ptr = nullptr;
  auto status = VarManager::Instance(session_id)->GetVarAddr(var_name, tensor_desc, &dev_ptr);
  if (status != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get Var add by name:%s failed, session_id:%lu", var_name.c_str(), session_id);
    GELOGE(status, "[Get][VarAddr] by name:%s failed, session_id:%lu", var_name.c_str(), session_id);
    return status;
  }
  int64_t offset = static_cast<int64_t>(reinterpret_cast<intptr_t>(dev_ptr));
  // logic_base_addr
  auto logic_var_base = VarManager::Instance(session_id)->GetVarMemLogicBase();
  // devcice_addr
  uint8_t *variable_addr = static_cast<uint8_t *>(var_mem_base + offset - logic_var_base);

  GE_CHK_RT_RET(rtMemcpy(dest, sizeof(int64_t), variable_addr, sizeof(int64_t), RT_MEMCPY_DEVICE_TO_HOST));
  return SUCCESS;
}

///
/// @brief insert EndOfSequence after GetNext
///
/// @param pre_node
/// @param graph
/// @return ge::NodePtr
///
ge::NodePtr IteratorOpPass::InsertEndOfSequenceNode(const ge::NodePtr &pre_node, const ge::NodePtr &memcpy_node,
                                                    const ge::ComputeGraphPtr &graph) {
  GELOGI("Start to insert EndOfSequence node.");
  GE_CHK_BOOL_EXEC(pre_node != nullptr, GELOGW("Pre node is null."); return nullptr);
  GE_CHK_BOOL_EXEC(graph != nullptr, GELOGW("graph is null."); return nullptr);
  ge::OpDescPtr end_of_seq_op_desc = CreateEndOfSequenceOp(pre_node);
  GE_CHK_BOOL_EXEC(end_of_seq_op_desc != nullptr, GELOGW("Create EndOfSequence op fail."); return nullptr);
  ge::NodePtr end_of_seq_node = graph->AddNode(end_of_seq_op_desc);
  GE_CHK_BOOL_EXEC(end_of_seq_node != nullptr, return nullptr,
                   "[Add][Node] %s to graph:%s failed.", end_of_seq_op_desc->GetName().c_str(),
                   graph->GetName().c_str());

  // getnext(data) --> EOS
  GE_CHK_BOOL_EXEC(pre_node->GetAllOutDataAnchorsSize() != 0, GELOGW("Pre node has no output."); return nullptr);
  auto out_anchor = pre_node->GetOutDataAnchor(0);
  ge::graphStatus status;
  status = GraphUtils::AddEdge(out_anchor, end_of_seq_node->GetInDataAnchor(0));
  GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                                     pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                                     end_of_seq_node->GetName().c_str(), end_of_seq_node->GetType().c_str());
                   return nullptr,
                   "[Add][Edge] between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                   pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                   end_of_seq_node->GetName().c_str(), end_of_seq_node->GetType().c_str());
  // EOS(control) --> subsequent of memcpy
  OutControlAnchorPtr out_ctrl_anchor = end_of_seq_node->GetOutControlAnchor();
  GE_CHK_BOOL_EXEC(out_ctrl_anchor != nullptr, GELOGW("out_ctrl_anchor is null."); return nullptr);
  // add ctrl edge
  for (const auto &out_node : memcpy_node->GetOutNodes()) {
    auto in_ctrl_anchor = out_node->GetInControlAnchor();
    if (in_ctrl_anchor == nullptr) {
      continue;
    }
    status = GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                       end_of_seq_node->GetName().c_str(), end_of_seq_node->GetType().c_str(),
                                       out_node->GetName().c_str(), out_node->GetType().c_str());
                     return nullptr,
                     "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                     end_of_seq_node->GetName().c_str(), end_of_seq_node->GetType().c_str(),
                     out_node->GetName().c_str(), out_node->GetType().c_str());
    GELOGI("Graph add EndOfSequence op out ctrl edge, dst node: %s.",
           out_node->GetName().c_str());
  }

  return end_of_seq_node;
}

///
/// @brief create EndOfSequence
///
/// @param pre_node
/// @return ge::OpDescPtr
///
ge::OpDescPtr IteratorOpPass::CreateEndOfSequenceOp(const ge::NodePtr &pre_node) {
  GELOGI("Start to create endOfSequence op.");
  GE_CHK_BOOL_EXEC(pre_node != nullptr,
                   REPORT_INNER_ERROR("E19999", "Param pre_node is nullptr, check invalid");
                   return nullptr, "[Check][Param] Input param pre_node invalid.");

  string node_name = pre_node->GetName() + "_EndOfSequence";
  ge::OpDescPtr op_desc = MakeShared<OpDesc>(node_name, ENDOFSEQUENCE);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed.");
    return op_desc;
  }
  ge::OpDescPtr pre_node_op_desc = pre_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_node_op_desc != nullptr,
                   REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
                   return nullptr, "[Get][OpDesc] failed, OpDesc of pre_node is invalid.");

  GELOGI("Create EndOfSequence op:%s.", op_desc->GetName().c_str());
  GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_node_op_desc->GetOutputDesc(0)) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "[Add][InputDesc] to op:%s(%s) failed", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return op_desc;
}

///
/// @brief insert memcpy after GetNext
///
/// @param pre_node
/// @param graph
/// @return ge::NodePtr
///
ge::NodePtr IteratorOpPass::InsertMemcpyAsyncNode(const ge::NodePtr &pre_node, const ge::ComputeGraphPtr &graph) {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, GELOGW("Pre node is null."); return nullptr);
  GE_CHK_BOOL_EXEC(graph != nullptr, GELOGW("graph is null."); return nullptr);
  ge::OpDescPtr memcpy_async_op_desc = CreateMemcpyAsyncOp(pre_node);
  GE_CHK_BOOL_EXEC(memcpy_async_op_desc != nullptr, GELOGW("Create memcpyAsync op fail."); return nullptr);
  ge::NodePtr memcpy_async_node = graph->AddNode(memcpy_async_op_desc);
  GE_CHK_BOOL_EXEC(memcpy_async_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                     memcpy_async_op_desc->GetName().c_str(), memcpy_async_op_desc->GetType().c_str(),
                                     graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed", memcpy_async_op_desc->GetName().c_str(),
                   memcpy_async_op_desc->GetType().c_str(), graph->GetName().c_str());

  // Data out
  for (auto &out_anchor : pre_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    ge::graphStatus status;
    GELOGI("Graph add memcpyAsync op in edge, index:%d.", out_anchor->GetIdx());
    for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_anchor == nullptr, GELOGW("peer_in_anchor is nullptr"); return nullptr);
      status = GraphUtils::RemoveEdge(out_anchor, peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999",
                                         "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                         pre_node->GetName().c_str(),
                                         pre_node->GetType().c_str(), out_anchor->GetIdx(),
                                         peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                                         peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                                         peer_in_anchor->GetIdx());
                       return nullptr,
                       "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                       pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                       peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                       peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                       peer_in_anchor->GetIdx());
      status = GraphUtils::AddEdge(memcpy_async_node->GetOutDataAnchor(out_anchor->GetIdx()), peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999",
                                         "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                         memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                         out_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                                         peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
                       return nullptr,
                       "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                       out_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                       peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
      GELOGI("Graph add memcpyAsync op out edge, src index:%d, dst index:%d, dst node: %s.", out_anchor->GetIdx(),
             peer_in_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
    }
    status = GraphUtils::AddEdge(out_anchor, memcpy_async_node->GetInDataAnchor(out_anchor->GetIdx()));
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999",
                                       "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                       pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                       out_anchor->GetIdx());
                     return nullptr,
                     "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                     pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                     memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                     out_anchor->GetIdx());
  }
  // Control out
  OutControlAnchorPtr out_ctrl_anchor = pre_node->GetOutControlAnchor();
  GE_IF_BOOL_EXEC(out_ctrl_anchor != nullptr,
      for (auto &peer_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
    ge::graphStatus status = GraphUtils::RemoveEdge(out_ctrl_anchor, peer_in_ctrl_anchor);
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999",
                                       "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                                       pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
                     return nullptr,
                     "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                     pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    status = GraphUtils::AddEdge(memcpy_async_node->GetOutControlAnchor(), peer_in_ctrl_anchor);
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
                     return nullptr,
                     "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                     memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    GELOGI("Graph add memcpyAsync op out ctrl edge, dst node: %s.",
           peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
  });
  GELOGI("Insert memcpyAsync op success.");

  return memcpy_async_node;
}

///
/// @brief create memcpy
///
/// @param pre_node
/// @return ge::OpDescPtr
///
ge::OpDescPtr IteratorOpPass::CreateMemcpyAsyncOp(const ge::NodePtr &pre_node) {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, return nullptr, "Input param invalid.");

  string node_name = pre_node->GetName() + "_MemcpyAsync";
  ge::OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), MEMCPYASYNC);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return op_desc;
  }
  GELOGI("Create memcpyAsync op:%s.", op_desc->GetName().c_str());

  ge::OpDescPtr pre_node_op_desc = pre_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_node_op_desc != nullptr,
                   REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
                   return nullptr, "[Get][OpDesc] failed, OpDesc of pre_node is invalid.");

  size_t out_size = pre_node_op_desc->GetOutputsSize();
  GELOGI("Create memcpyAsync op, pre_node out_size: %zu.", out_size);
  for (size_t i = 0; i < out_size; i++) {
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                       pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
                     return nullptr,
                     "[Add][InputDesc] to op:%s(%s) failed",
                     pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                       pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
                     return nullptr,
                     "[Add][OutputDesc] to op:%s(%s) failed",
                     pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
  }

  return op_desc;
}

Status IteratorOpPass::SetRtContext(uint64_t session_id, uint32_t graph_id, rtContext_t rt_context, rtCtxMode_t mode) {
  GELOGI("set rt_context, session id: %lu, graph id: %u, mode %d, device id:%u.", session_id,
         graph_id, static_cast<int>(mode), ge::GetContext().DeviceId());

  GE_CHK_RT_RET(rtCtxCreate(&rt_context, mode, ge::GetContext().DeviceId()));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddRtContext(session_id, graph_id, rt_context);

  return SUCCESS;
}
}  // namespace ge
