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

#include "hybrid/model/hybrid_model_builder.h"
#include <algorithm>
#include "common/math/math_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/utils/graph_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
using domi::LogTimeStampDef;
using domi::TaskDef;
namespace {
const uint32_t kSubgraphIndex = 0U;
const uint32_t kVarOutputIndex = 0U;
const uint64_t kProfilingFpStartLogid = 1U;
const uint64_t kProfilingBpEndLogid = 2U;
const uint64_t kProfilingIterEndLogid = 65535U;
const int kBytes = 8;
const int kDecimal = 10;
const uint8_t kLoopEnterIdx = 0;
const uint8_t kLoopIterationIdx = 1;
const uint8_t kLoopMergeSize = 2;
const uint8_t kStreamSwitchIdx = 1;
const uint8_t kStreamSwitchNum = 2;
const uint32_t kStringHeadElems = 2;
const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
const char *const kProfilingGraph = "ProfilingGraph";
const char *const kProfilingFpNode = "ProfilingFpNode";
const char *const kProfilingBpNode = "ProfilingBpNode";
const char *const kProfilingEndNode = "ProfilingEndNode";
const char *const kProfilingArNode = "ProfilingAllReduceNode";
const char *const kEngineNameRts = "DNN_VM_RTS_OP_STORE";
const char *const kForceInfershape = "_force_infershape_when_running";

const std::set<std::string> kExecutionDependentTypes{ IF, STATELESSIF, CASE, STREAMSWITCH };
const std::set<std::string> kMergeInputSkipTypes{ STREAMACTIVE, STREAMSWITCH, CONSTANT, CONSTANTOP };
const std::set<std::string> kStreamActiveTypes{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };

Status SetOutputNameAttr(ComputeGraph &graph) {
  vector<string> output_names;
  for (const auto &node : graph.GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto op_type = op_desc->GetType();
    if (op_type == NETOUTPUT) {
      for (InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
        const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
        NodePtr in_node = peer_out_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(in_node);
        output_names.push_back(in_node->GetName());
      }
    }
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(&graph, ATTR_MODEL_OUT_NODES_NAME, output_names),
      GELOGE(FAILED, "[Invoke][SetListStr] failed, graph:%s name:%s.", graph.GetName().c_str(),
             ATTR_MODEL_OUT_NODES_NAME.c_str());
      REPORT_CALL_ERROR("E19999", "SetListStr failed, graph:%s name:%s.",  graph.GetName().c_str(),
                        ATTR_MODEL_OUT_NODES_NAME.c_str());
      return FAILED);
  return SUCCESS;
}

int64_t CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int64_t var_size = 0;
  auto data_type = desc.GetDataType();
  if (data_type == DT_STRING) {
    (void) TensorUtils::GetSize(desc, var_size);
    return var_size;
  }

  if (TensorUtils::GetTensorMemorySizeInBytes(desc, var_size) != GRAPH_SUCCESS) {
    GELOGW("Failed to calc var data size");
    return -1;
  }

  return var_size;
}

Status CollectDependenciesForFusedGraph(NodeItem &node_item, std::set<OpDesc *> &data_ops) {
  for (const auto &node : node_item.fused_subgraph->nodes) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &depends = op_desc->GetOpInferDepends();
    if (depends.empty()) {
      continue;
    }

    for (auto &input_name : depends) {
      auto input_index = op_desc->GetInputIndexByName(input_name);
      auto src_node = NodeUtils::GetInDataNodeByIndex(*node, input_index);
      GE_CHECK_NOTNULL(src_node);
      auto src_op_desc = src_node->GetOpDesc();
      GE_CHECK_NOTNULL(src_op_desc);
      if (src_node->GetType() != DATA_TYPE) {
        GELOGE(UNSUPPORTED, "[Check][NodeType][%s::%s] Node in fused subgraph can only depend on Data nodes,"
               "but depend on %s actually", node_item.NodeName().c_str(), node->GetName().c_str(),
               src_node->GetType().c_str());
        REPORT_INNER_ERROR("E19999", "[%s::%s] Node in fused subgraph can only depend on Data nodes,"
                           "but depend on %s actually.", node_item.NodeName().c_str(), node->GetName().c_str(),
                           src_node->GetType().c_str());
        return UNSUPPORTED;
      }

      data_ops.emplace(src_op_desc.get());
    }
  }

  return SUCCESS;
}
}  // namespace
HybridModelBuilder::HybridModelBuilder(HybridModel &hybrid_model)
    : hybrid_model_(hybrid_model), runtime_param_(hybrid_model.root_runtime_param_) {
  ge_root_model_ = hybrid_model_.ge_root_model_;
}

Status HybridModelBuilder::Build() {
  GE_CHK_STATUS_RET(ValidateParams(), "[Invoke][ValidateParams] failed, model_name_:[%s]", GetGraphName());
  hybrid_model_.model_name_ = ge_root_model_->GetModelName();
  GELOGI("[%s] Start to build hybrid model.", GetGraphName());
  GE_CHK_STATUS_RET(CopyGraph(), "[Invoke][CopyGraph] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitRuntimeParams(), "[Invoke][InitRuntimeParams] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(RecoverGraphUnknownFlag(),
                    "[Invoke][RecoverGraphUnknownFlag] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(IndexSpecialNodes(), "[Invoke][IndexSpecialNodes] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(IndexTaskDefs(), "[Invoke][IndexTaskDefs] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitWeights(), "[Invoke][InitWeights] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(LoadGraph(), "[Invoke][LoadGraph] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(AssignUninitializedConstantOps(),
                    "[Invoke][AssignUninitializedConstantOps] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(TransAllVarData(), "[Invoke][TransAllVarData] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(CopyVarData(), "[Invoke][CopyVarData] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitModelMem(), "[Invoke][InitModelMem] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitConstantOps(), "[Invoke][InitConstantOps] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitVariableTensors(), "[Invoke][InitVariableTensors], model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(LoadTasks(), "[Invoke][LoadTasks] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(OptimizeDependenciesForConstantInputs(),
                    "[Invoke][OptimizeDependenciesForConstantInputs] failed, model_name_:[%s]",
                    GetGraphName());
  GELOGI("[%s] Done building hybrid model successfully.", GetGraphName());
  return SUCCESS;
}

Status HybridModelBuilder::BuildForSingleOp() {
  GE_CHK_STATUS_RET(ValidateParams(), "[Invoke][ValidateParams] failed, model_name_:[%s]", GetGraphName());
  hybrid_model_.root_graph_ = ge_root_model_->GetRootGraph();
  hybrid_model_.model_name_ = ge_root_model_->GetRootGraph()->GetName();
  GELOGI("[%s] Start to build hybrid model.", GetGraphName());
  auto ret = ge_root_model_->GetSubgraphInstanceNameToModel();
  const GeModelPtr ge_model = ret[hybrid_model_.root_graph_->GetName()];
  GE_CHK_STATUS_RET(IndexTaskDefs(hybrid_model_.root_graph_, ge_model),
                    "[Invoke][IndexTaskDefs] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(LoadGraph(), "[Invoke][LoadGraph] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(InitWeights(), "[Invoke][InitWeights] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET(LoadTasks(), "[Invoke][LoadTasks] failed, model_name_:[%s]", GetGraphName());
  GELOGI("[%s] Done building hybrid model for single op successfully.", GetGraphName());
  return SUCCESS;
}

Status HybridModelBuilder::ValidateParams() {
  GE_CHECK_NOTNULL(ge_root_model_);
  GE_CHECK_NOTNULL(ge_root_model_->GetRootGraph());
  return SUCCESS;
}

Status HybridModelBuilder::CopyGraph() {
  GELOGD("Copy compute graph begin.");
  auto root_graph = ge_root_model_->GetRootGraph();

  std::string new_graph_name = ge_root_model_->GetRootGraph()->GetName();
  ComputeGraphPtr new_root_graph = MakeShared<ComputeGraph>(new_graph_name);
  GE_CHECK_NOTNULL(new_root_graph);
  int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  graphStatus ret = GraphUtils::CopyComputeGraph(root_graph, new_root_graph, node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Copy compute graph failed.");
    return GRAPH_FAILED;
  }
  hybrid_model_.root_graph_ = new_root_graph;

  GELOGD("Copy compute graph[%s] success.", new_graph_name.c_str());
  return SUCCESS;
}

Status HybridModelBuilder::BuildNodeItem(const NodePtr &node, NodeItem &node_item) {
  auto op_desc = node->GetOpDesc();
  GE_CHK_STATUS_RET(ParseForceInfershapeNodes(node, node_item),
                    "[Invoke][ParseForceInfershapeNodes]failed, node:[%s].",
                    node_item.NodeName().c_str());
  vector<string> dependencies = node->GetOpDesc()->GetOpInferDepends();
  GE_CHK_STATUS_RET(ParseDependentInputNodes(node_item, dependencies),
                    "[Invoke][ParseDependentInputNodes]failed, node:[%s].",
                    node_item.NodeName().c_str());

  node_item.outputs.resize(node_item.num_outputs);
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto out_data_anchor = node->GetOutDataAnchor(i);
    if (out_data_anchor == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Get][OutDataAnchor]out anchor[%d] of node %s is nullptr", i, node->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "out anchor[%d] of node %s is nullptr.", i, node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    for (auto &dst_in_anchor: out_data_anchor->GetPeerInDataAnchors()) {
      auto dst_node = dst_in_anchor->GetOwnerNode();
      if (dst_node == nullptr) {
        GELOGW("dst node is nullptr. out anchor = %d", out_data_anchor->GetIdx());
        continue;
      }

      NodeItem *dst_node_item = nullptr;
      GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                        "[GetOrCreate][NodeItem] failed, dst_node:[%s].", dst_node->GetName().c_str());
      int canonical_index;
      GE_CHK_STATUS_RET(dst_node_item->GetCanonicalInputIndex(dst_in_anchor->GetIdx(), canonical_index),
                        "[Invoke][GetCanonicalInputIndex] failed, dst_node:[%s].", dst_node->GetName().c_str());

      node_item.outputs[i].emplace_back(canonical_index, dst_node_item);
      node_item.SetDataSend(dst_node_item, dst_in_anchor->GetIdx());
    }
  }

  GE_CHK_STATUS_RET_NOLOG(ResolveRefIo(node_item));
  return SUCCESS;
}

Status HybridModelBuilder::ResolveRefIo(NodeItem &node_item) {
  bool is_ref = false;
  auto &op_desc = *node_item.op_desc;
  (void) AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (!is_ref) {
    return SUCCESS;
  }

  auto inputs = op_desc.GetAllInputName();
  auto outputs = op_desc.GetAllOutputName();
  for (auto &output : outputs) {
    for (auto &input : inputs) {
      if (input.first == output.first) {
        int input_idx;
        GE_CHK_STATUS_RET_NOLOG(node_item.GetCanonicalInputIndex(input.second, input_idx));
        auto output_idx = static_cast<int>(output.second);
        node_item.reuse_inputs[output_idx] = input_idx;
        GELOGD("[%s] Output[%d] reuse input[%d]", node_item.NodeName().c_str(), output_idx, input_idx);
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::GetOrCreateNodeItem(const NodePtr &node, NodeItem **node_item) {
  auto &node_items = hybrid_model_.node_items_;
  auto it = node_items.find(node);
  if (it != node_items.end()) {
    *node_item = it->second.get();
    return SUCCESS;
  }

  std::unique_ptr<NodeItem> new_node;
  GE_CHK_STATUS_RET(NodeItem::Create(node, new_node), "[Invoke][Create] failed, model_name_:[%s]", GetGraphName());
  GE_CHK_STATUS_RET_NOLOG(NodeExecutorManager::GetInstance().GetExecutor(*node, &new_node->node_executor));

  // we do not need L2 Buffer
  const char *const kIsFirstNode = "is_first_node";
  const char *const kIsLastNode = "is_last_node";
  (void) AttrUtils::SetBool(new_node->op_desc, kIsFirstNode, false);
  (void) AttrUtils::SetBool(new_node->op_desc, kIsLastNode, false);

  new_node->node_id = static_cast<int>(new_node->op_desc->GetId());
  NodeExecutorManager::ExecutorType executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*node);
  new_node->is_profiling_report = (executor_type == NodeExecutorManager::ExecutorType::AICORE) ||
                                  (executor_type == NodeExecutorManager::ExecutorType::AICPU_TF) ||
                                  (executor_type == NodeExecutorManager::ExecutorType::AICPU_CUSTOM);
  *node_item = new_node.get();
  node_items[node] = std::move(new_node);
  return SUCCESS;
}

Status HybridModelBuilder::ParseForceInfershapeNodes(const NodePtr &node, NodeItem &node_item) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // not care result, if no this attr, stand for the op does not need force infershape
  (void) AttrUtils::GetBool(op_desc, kForceInfershape, node_item.is_need_force_infershape);
  GELOGD("node [%s] is need do infershape, flag is %d",
         op_desc->GetName().c_str(),
         node_item.is_need_force_infershape);
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependencies(NodeItem &node_item, const std::vector<string> &dependencies,
                                             std::set<NodePtr> &dependent_for_shape_inference) {
  for (const auto &input_name : dependencies) {
    int input_index = node_item.op_desc->GetInputIndexByName(input_name);
    if (input_index < 0) {
      GELOGE(INTERNAL_ERROR, "[Get][InputIndex]failed, node:[%s] inputname: %s.",
             node_item.NodeName().c_str(), input_name.c_str());
      REPORT_CALL_ERROR("E19999", "GetInputIndexByName failed, node:[%s] inputname: %s.",
                        node_item.NodeName().c_str(), input_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &in_anchor = node_item.node->GetInDataAnchor(input_index);
    GE_CHECK_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    if (src_node_item->NodeType() == DATA) {
      auto op_desc = src_node_item->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      auto tensor = op_desc->MutableInputDesc(0);
      if (AttrUtils::HasAttr(tensor, ATTR_NAME_VALUE)) {
        GELOGD("Skip d2h memcpy, get hostmem from node %s.", src_node_item->NodeName().c_str());
        continue;
      }
    }
    src_node_item->to_const_output_id_list.emplace(peer_out_anchor->GetIdx());
    dependent_for_shape_inference.emplace(src_node);
    host_input_value_dependencies_[&node_item].emplace_back(peer_out_anchor->GetIdx(), src_node_item);
    GELOGD("[%s] Dependent added from output of [%s:%d]",
           node_item.NodeName().c_str(),
           src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx());
  }
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentInputNodes(NodeItem &node_item, const std::vector<string> &dependencies) {
  std::set<NodePtr> dependent_for_shape_inference;
  std::set<NodePtr> dependent_for_execution;
  auto &ge_node = node_item.node;
  bool is_hccl_op = node_item.IsHcclOp();

  // The input tensors become valid after computation is done for parent nodes of type DEPEND_COMPUTE.
  // Wait for these parent nodes before execution.
  for (const auto &in_anchor : ge_node->GetAllInDataAnchors()) {
    const auto &peer_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      GELOGD("[%s] Input[%d] do not have peer anchor", node_item.NodeName().c_str(), in_anchor->GetIdx());
      continue;
    }
    auto src_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, &src_node_item),
                      "[%s] failed to get or create node item", src_node->GetName().c_str());

    if (src_node_item->shape_inference_type == DEPEND_COMPUTE || is_hccl_op || src_node_item->IsHcclOp()) {
      GELOGD("[%s](%s) Add input data dependent node [%s](%s), shape inference type = %d",
             ge_node->GetName().c_str(),
             ge_node->GetType().c_str(),
             src_node->GetName().c_str(),
             src_node->GetType().c_str(),
             static_cast<int>(src_node_item->shape_inference_type));
      src_node_item->has_observer = true;
      dependent_for_execution.emplace(src_node);
    }

    if (src_node_item->shape_inference_type == DEPEND_SHAPE_RANGE) {
      GELOGD("[%s] Add input shape dependent node [%s] due to inference type = DEPEND_SHAPE_RANGE",
             node_item.NodeName().c_str(),
             src_node_item->NodeName().c_str());
      src_node_item->has_observer = true;
      dependent_for_shape_inference.emplace(src_node);
    }
  }

  if (node_item.node_type == NETOUTPUT) {
    for (const auto &src_node : ge_node->GetInControlNodes()) {
      auto src_node_item = MutableNodeItem(src_node);
      if ((src_node_item != nullptr) && src_node_item->IsHcclOp()) {
        GELOGD("[%s](%s) Add input control dependent node [%s](%s)",
               ge_node->GetName().c_str(),
               ge_node->GetType().c_str(),
               src_node->GetName().c_str(),
               src_node->GetType().c_str());
        dependent_for_execution.emplace(src_node);
      }
    }
  }

  // cond or branch need to be prepared before the execution of IF or CASE
  if (kExecutionDependentTypes.count(node_item.node_type) > 0) {
    auto src_node = NodeUtils::GetInDataNodeByIndex(*ge_node, 0); // cond input
    GE_CHECK_NOTNULL(src_node);
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    dependent_for_execution.emplace(src_node);
    GELOGD("[%s] Dependent added from %s for control op's cond/branch",
           node_item.NodeName().c_str(),
           src_node_item->NodeName().c_str());
  }

  GE_CHK_STATUS_RET(ParseDependencies(node_item, dependencies, dependent_for_shape_inference));

  GE_CHK_STATUS_RET(ParseDependentForFusedSubgraph(node_item, dependent_for_shape_inference));
  for (const auto &dep_node : dependent_for_shape_inference) {
    auto src_node_item = MutableNodeItem(dep_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->has_observer = true;
    node_item.dependents_for_shape_inference.emplace_back(dep_node);
  }

  for (const auto &dep_node : dependent_for_execution) {
    auto src_node_item = MutableNodeItem(dep_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->has_observer = true;
    node_item.dependents_for_execution.emplace_back(dep_node);
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentForFusedSubgraph(NodeItem &node_item, std::set<ge::NodePtr> &dependencies) {
  if (node_item.fused_subgraph == nullptr) {
    return SUCCESS;
  }

  std::set<OpDesc *> data_ops;
  GE_CHK_STATUS_RET_NOLOG(CollectDependenciesForFusedGraph(node_item, data_ops));
  for (auto &op_desc : data_ops) {
    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(*op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(INTERNAL_ERROR, "[Invoke][GetInt] failed, node:[%s]  attr:[%s]",
             op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_CALL_ERROR("E19999", "invoke GetInt failed, node:[%s]  attr:[%s]",
                        op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return INTERNAL_ERROR;
    }

    const auto &in_anchor = node_item.node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(src_node, &src_node_item));
    op_desc->SetId(src_node_item->op_desc->GetId());
    GELOGD("[%s::%s] Node id was set to that of outer src node's, src_node = %s",
           node_item.NodeName().c_str(),
           op_desc->GetName().c_str(),
           src_node_item->NodeName().c_str());
    src_node_item->to_const_output_id_list.emplace(peer_out_anchor->GetIdx());
    dependencies.emplace(src_node);
    GELOGD("[%s] Dependent added from output of [%s:%d]",
           node_item.NodeName().c_str(),
           src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx());
  }

  return SUCCESS;
}

Status HybridModelBuilder::UpdateAnchorStatus(const NodePtr &node) {
  if (NodeUtils::SetAllAnchorStatus(node) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Invoke][SetAllAnchorStatus] failed, node:[%s].", node->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "[%s] NodeUtils::SetAllAnchorStatus failed.", node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  for (auto &anchor : node->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_SUSPEND, node:[%s].",
               node->GetName().c_str());
        REPORT_CALL_ERROR("E19999", "SetStatus failed to set ANCHOR_SUSPEND, node:[%s].", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    } else if (peer_anchor->GetOwnerNode()->GetType() == CONSTANT) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_CONST) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_CONST, node:[%s].", node->GetName().c_str());
        REPORT_CALL_ERROR("E19999", "SetStatus failed to set ANCHOR_CONST, node:[%s].", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    } else {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_DATA) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_DATA, node:[%s].", node->GetName().c_str());
        REPORT_CALL_ERROR("E19999", "SetStatus failed to set ANCHOR_DATA, node:[%s].", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::DoUnlinkDataAnchors(const OutDataAnchorPtr &out_data_anchor,
                                               const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->Unlink(in_data_anchor),
                          "[Invoke][Unlink] failed to unlink %s:%d from %s:%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(), out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in unlinking %s:%d from %s:%d",
         out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(),
         in_data_anchor->GetOwnerNode()->GetName().c_str(),
         in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::DoLinkDataAnchors(OutDataAnchorPtr &out_data_anchor, InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->LinkTo(in_data_anchor), "[Invoke][LinkTo]Failed to link %s:%d to %s:%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(),
                          out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(),
                          in_data_anchor->GetIdx());

  GELOGD("Succeeded in linking %s:%d to %s:%d",
         out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(),
         in_data_anchor->GetOwnerNode()->GetName().c_str(),
         in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::MergeInputNodes(ComputeGraph &graph) {
  const auto &wrapped_node = graph.GetParentNode();
  std::set<NodePtr> root_nodes;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != DATA_TYPE) {
      if (node->GetInDataNodes().empty()) {
        root_nodes.emplace(node);
      }

      continue;
    }

    auto data_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(data_op_desc);

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Invoke][GetInt] failed, node:[%s] attr:[%s]",
             data_op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_CALL_ERROR("E19999", "GetInt failed, node:[%s] attr:[%s]",
                        data_op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return FAILED;
    }

    auto wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(wrapped_node_in_anchor);
    auto src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    wrapped_node_in_anchor->UnlinkAll();

    // link src to outputs of DataNode
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_data_anchor);
      for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_data_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto in_nodes = dst_node->GetInDataNodes();
        if (std::all_of(in_nodes.begin(), in_nodes.end(), [](const NodePtr &n) { return n->GetType() == DATA; })) {
          root_nodes.emplace(dst_node);
        }
        GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(out_data_anchor, peer_in_data_anchor));
        GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, peer_in_data_anchor));
      }
    }
  }

  // transfer in control edges to all root nodes
  for (auto &root_node : root_nodes) {
    auto in_nodes = root_node->GetInAllNodes();
    std::set<NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (auto &in_control_node : wrapped_node->GetInControlNodes()) {
      if (in_node_set.count(in_control_node) == 0 && kMergeInputSkipTypes.count(root_node->GetType()) == 0) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), root_node->GetName().c_str());
        GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
        (void) in_control_node->GetOutControlAnchor()->LinkTo(root_node->GetInControlAnchor());
      }
    }
  }

  wrapped_node->GetInControlAnchor()->UnlinkAll();
  return SUCCESS;
}

Status HybridModelBuilder::MergeNetOutputNode(ComputeGraph &graph) {
  const auto &parent_node = graph.GetParentNode();
  const NodePtr &net_output_node = graph.FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("Graph has no netoutput no need to merge");
    return SUCCESS;
  }
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  auto all_in_nodes = net_output_node->GetInAllNodes();
  auto all_out_nodes = parent_node->GetOutAllNodes();
  net_output_node->GetInControlAnchor()->UnlinkAll();
  parent_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNode());
    GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(src_out_anchor, in_data_anchor));

    auto index = in_data_anchor->GetIdx();
    auto input_desc = net_output_desc->MutableInputDesc(index);
    if (input_desc == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Invoke][MutableInputDesc][%s] Failed to get input desc[%d]",
             net_output_desc->GetName().c_str(), index);
      REPORT_CALL_ERROR("E19999", "[%s] Failed to get input desc[%d].", net_output_desc->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }

    uint32_t parent_index = 0;
    if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found.",
             graph.GetName().c_str(), index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    const OutDataAnchorPtr &parent_out_anchor = parent_node->GetOutDataAnchor(parent_index);
    GE_CHECK_NOTNULL(parent_out_anchor);
    for (InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
      if (dst_in_anchor == nullptr) {
        continue;
      }

      GE_CHECK_NOTNULL(dst_in_anchor->GetOwnerNode());
      GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(parent_out_anchor, dst_in_anchor));
      GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, dst_in_anchor));
    }
  }

  // transfer out control edges
  std::set<NodePtr> in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  std::set<NodePtr> out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    auto out_nodes = src_node->GetOutAllNodes();
    std::set<NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0) {
        src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraphs(ComputeGraphPtr &root_graph, ComputeGraphPtr &merged_graph) {
  merged_graph = MakeShared<ComputeGraph>("MergedGraph");
  merged_graph->SetGraphUnknownFlag(root_graph->GetGraphUnknownFlag());
  for (const auto &node : root_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto &op_type = node->GetType();
    if (op_type != PARTITIONEDCALL) {
      merged_graph->AddNode(node);
      GELOGD("[%s] Node added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    auto subgraph = NodeUtils::GetSubgraph(*node, kSubgraphIndex);
    GE_CHECK_NOTNULL(subgraph);
    bool is_unknown_shape = subgraph->GetGraphUnknownFlag();
    if (!is_unknown_shape) {
      merged_graph->AddNode(node);
      GELOGD("[%s] Known shape partitioned call added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
      uint32_t stage_level = UINT32_MAX;
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level)) {
        for (const auto &stage_node : subgraph->GetAllNodes()) {
          GELOGD("Set ATTR_STAGE_LEVEL on node %s, stage_level=%u", stage_node->GetName().c_str(), stage_level);
          (void)AttrUtils::SetInt(stage_node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
        }
      }
    }
    GE_CHK_GRAPH_STATUS_RET(UnfoldSubgraph(root_graph, merged_graph, *subgraph),
                            "[Invoke][UnfoldSubgraph][%s] Failed to merge subgraph.",
                            subgraph->GetName().c_str());
  }

  // invoke before adding subgraphs. in case modify node id in known-shaped subgraphs.
  GE_CHK_GRAPH_STATUS_RET(merged_graph->TopologicalSorting(),
                          "[Invoke][TopologicalSorting]Failed to invoke TopologicalSorting on merged graph.");
  GE_DUMP(merged_graph, "hybrid_merged_graph_BeforeStageSort");
  merged_graph->TopologicalSorting([](const NodePtr &a, const NodePtr &b) -> bool {
    uint32_t a_level = UINT32_MAX;
    (void)AttrUtils::GetInt(a->GetOpDesc(), ATTR_STAGE_LEVEL, a_level);
    uint32_t b_level = UINT32_MAX;
    (void)AttrUtils::GetInt(b->GetOpDesc(), ATTR_STAGE_LEVEL, b_level);
    return a_level < b_level;
  });

  for (auto &remained_subgraph : root_graph->GetAllSubgraphs()) {
    GELOGD("Adding subgraph [%s] to merged-graph.", remained_subgraph->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(merged_graph->AddSubgraph(remained_subgraph),
                            "[Invoke][AddSubgraph]Failed to add subgraph [%s]",
                            remained_subgraph->GetName().c_str());
    remained_subgraph->SetParentGraph(merged_graph);
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraph(ComputeGraphPtr &root_graph,
                                          ComputeGraphPtr &parent_graph,
                                          ComputeGraph &sub_graph) {
  auto parent_node = sub_graph.GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  GE_CHK_STATUS_RET(MergeInputNodes(sub_graph),
                    "[Invoke][MergeInputNodes][%s] Failed to merge data nodes for subgraph",
                    sub_graph.GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(sub_graph),
                    "[Invoke][MergeNetOutputNode][%s] Failed to merge net output nodes for subgraph",
                    sub_graph.GetName().c_str());
  GELOGD("[%s] Done merging subgraph inputs and outputs successfully", sub_graph.GetName().c_str());

  for (auto &sub_node : sub_graph.GetDirectNode()) {
    auto sub_op_type = sub_node->GetType();
    if (sub_op_type == DATA_TYPE || sub_op_type == NETOUTPUT) {
      continue;
    }
    if (sub_op_type == PARTITIONEDCALL) {
      auto sub_sub_graph = NodeUtils::GetSubgraph(*sub_node, kSubgraphIndex);
      GE_CHECK_NOTNULL(sub_sub_graph);
      if (sub_sub_graph->GetGraphUnknownFlag()) {
        GE_CHK_STATUS_RET(UnfoldSubgraph(root_graph, parent_graph, *sub_sub_graph),
                          "[Invoke][UnfoldSubgraph][%s] Failed to merge subgraph",
                          sub_sub_graph->GetName().c_str());
        continue;
      }
    }

    if (!sub_node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
      for (size_t i = 0; i < sub_node->GetOpDesc()->GetSubgraphInstanceNames().size(); ++i) {
        auto sub_sub_graph = NodeUtils::GetSubgraph(*sub_node, i);
        GE_CHECK_NOTNULL(sub_sub_graph);
        sub_sub_graph->SetParentGraph(parent_graph);
      }
    }
    parent_graph->AddNode(sub_node);
    GELOGD("[%s::%s] added to parent graph: [%s].",
           sub_graph.GetName().c_str(),
           sub_node->GetName().c_str(),
           parent_graph->GetName().c_str());
    sub_node->SetOwnerComputeGraph(parent_graph);
  }

  GELOGD("[%s] Done merging subgraph. remove it from root graph", sub_graph.GetName().c_str());
  root_graph->RemoveSubgraph(sub_graph.GetName());
  return SUCCESS;
}

Status HybridModelBuilder::BuildOutputMapping(GraphItem &graph_item,
                                              const NodeItem &node_item,
                                              bool is_root_graph) {
  auto output_size = node_item.num_inputs;
  graph_item.output_edges_.resize(output_size);

  for (auto &in_data_anchor : node_item.node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    auto src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);

    auto src_node_item = GetNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    auto output_idx = in_data_anchor->GetIdx();
    auto output_offset = src_node_item->output_start + peer_out_anchor->GetIdx();
    GELOGI("Output[%d], node = %s, output_index = %d, output_offset = %d ",
           output_idx,
           src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx(),
           output_offset);

    GE_CHECK_LE(output_idx, output_size - 1);
    graph_item.output_edges_[output_idx] = {src_node_item, peer_out_anchor->GetIdx()};
  }

  if (!is_root_graph) {
    for (uint32_t i = 0; i < static_cast<uint32_t>(output_size); ++i) {
      uint32_t p_index = i;
      // Net output of Subgraph of while do not have parent index
      if (AttrUtils::GetInt(node_item.op_desc->GetInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
        GELOGD("[%s] Parent index not set for input[%u].", node_item.NodeName().c_str(), i);
      }

      graph_item.output_index_mapping_.emplace_back(p_index);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGraph() {
  auto root_graph = hybrid_model_.root_graph_;
  if (!GetContext().GetHostExecFlag()) {
    std::shared_ptr<ComputeGraph> merged_graph;
    GELOGI("Before merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu",
           root_graph->GetDirectNodesSize(),
           root_graph->GetAllNodesSize());
    hybrid_model_.orig_root_graph_ = root_graph;
    GE_CHK_GRAPH_STATUS_RET(UnfoldSubgraphs(root_graph, merged_graph),
                            "[Invoke][UnfoldSubgraphs]Failed to unfold subgraphs, model_name_:%s.", GetGraphName());
    root_graph = std::move(merged_graph);
    GELOGI("After merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu",
           root_graph->GetDirectNodesSize(),
           root_graph->GetAllNodesSize());
  }

  hybrid_model_.root_graph_ = root_graph;
  GE_CHK_STATUS_RET(RelinkNextIteration(), "[%s] Relink NextIteration failed", GetGraphName());
  // Reset node id by topological order across all subgraphs
  int64_t index = 0;
  for (const auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto parent_graph = node->GetOwnerComputeGraph();
    // No need to update nodes in known subgraph
    if (parent_graph != nullptr && !parent_graph->GetGraphUnknownFlag()) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetId(index++);
  }
  GE_DUMP(root_graph, "hybrid_merged_graph");
  GE_CHK_STATUS_RET(LoadDynamicSubgraph(*root_graph, true),
                    "[Invoke][LoadDynamicSubgraph]Failed to load root graph, model_name_:%s.", GetGraphName());
  GELOGD("Done loading root graph successfully.");
  GE_CHK_STATUS_RET(hybrid_model_.root_graph_item_->GroupNodes(),
                    "[Invoke][GroupNodes]Failed to group nodes for root graph, model_name_:%s.", GetGraphName());

  for (auto &sub_graph : root_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    GELOGD("Start to load subgraph [%s]", sub_graph->GetName().c_str());
    auto parent_node = sub_graph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    auto parent_node_item = MutableNodeItem(parent_node);
    // parent node is in another known subgraph
    if (parent_node_item == nullptr) {
      GELOGD("[%s] Subgraph is in another known shaped subgraph, skip it.", sub_graph->GetName().c_str());
      continue;
    }

    if (sub_graph->GetGraphUnknownFlag()) {
      GE_CHK_STATUS_RET(LoadDynamicSubgraph(*sub_graph, false),
                        "[Invoke][LoadDynamicSubgraph]Failed to load subgraph: [%s]",
                        sub_graph->GetName().c_str());
    } else {
      // if parent is function control op. need add a virtual partitioned call
      if (parent_node_item->IsControlFlowV2Op()) {
        GE_CHK_STATUS_RET(LoadKnownShapedSubgraph(*sub_graph, parent_node_item),
                          "[Invoke][LoadKnownShapedSubgraph]Failed to load function control op subgraph [%s]",
                          sub_graph->GetName().c_str());
      }
    }
  }
  for (auto &it : hybrid_model_.known_shape_sub_models_) {
    auto node_item = MutableNodeItem(it.first);
    GE_CHECK_NOTNULL(node_item);
    AscendString graph_name;
    GE_CHK_GRAPH_STATUS_RET(it.second->GetGraph().GetName(graph_name), "Failed to get subgraph name");
    GE_CHECK_NOTNULL(graph_name.GetString());
    auto subgraph = hybrid_model_.GetRootGraph()->GetSubgraph(graph_name.GetString());
    GE_CHECK_NOTNULL(subgraph);
    GE_CHK_STATUS_RET(IdentifyVariableOutputs(*node_item, subgraph),
                      "[Invoke][IdentifyVariableOutputs][%s] Failed to identify ref outputs.",
                      node_item->NodeName().c_str());
  }
  GE_CHK_STATUS_RET(ParseDependentByParallelGroup(),
                    "[Invoke][ParseDependentByParallelGroup]Failed to establish dependencies for hccl ops,"
                    "model_name_:%s.", GetGraphName());
  GELOGI("Done loading all subgraphs successfully.");
  return SUCCESS;
}

const NodeItem *HybridModelBuilder::GetNodeItem(const NodePtr &node) const {
  return hybrid_model_.GetNodeItem(node);
}

NodeItem *HybridModelBuilder::MutableNodeItem(const NodePtr &node) {
  return hybrid_model_.MutableNodeItem(node);
}

Status HybridModelBuilder::VarNodeToTensor(const NodePtr &var_node, std::unique_ptr<TensorValue> &tensor) {
  string var_name = var_node->GetName();
  auto tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0);
  uint8_t *var_logic = nullptr;
  GE_CHK_STATUS_RET(var_manager_->GetVarAddr(var_name, *tensor_desc, &var_logic),
                    "[Invoke][GetVarAddr]Failed to get var addr. var_name = %s, session_id = %ld",
                    var_name.c_str(),
                    hybrid_model_.GetSessionId());

  rtMemType_t memory_type = RT_MEMORY_HBM;
  uint32_t mem_type = 0;
  if (AttrUtils::GetInt(var_node->GetOpDesc(), ATTR_OUTPUT_MEMORY_TYPE, mem_type) && (mem_type == 1)) {
    memory_type = RT_MEMORY_RDMA_HBM;
  }
  uint8_t *dev_mem = var_manager_->GetVarMemoryAddr(var_logic, memory_type);
  if (dev_mem == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][GetVarMemoryAddr]Failed to copy var %s from device,"
           "cant not get var addr from logic addr %p", var_node->GetName().c_str(), var_logic);
    REPORT_CALL_ERROR("E19999", "GetVarMemoryAddr failed, Failed to copy var %s from device,"
                      "cant not get var addr from logic addr %p", var_node->GetName().c_str(), var_logic);
    return INTERNAL_ERROR;
  }

  int64_t var_size = CalcVarSizeInBytes(*tensor_desc);
  GE_CHECK_GE(var_size, 0);
  tensor.reset(new(std::nothrow)TensorValue(dev_mem, static_cast<size_t>(var_size)));
  GE_CHECK_NOTNULL(tensor);
  GELOGI("Get var memory addr %p for node %s, size = %ld, mem_type=%u", dev_mem, var_name.c_str(), var_size, mem_type);
  return SUCCESS;
}

Status HybridModelBuilder::HandleDtString(const GeTensor &tensor, void *var_addr) {
  auto desc = tensor.GetTensorDesc();
  if (desc.GetDataType() == DT_STRING) {
    GeShape tensor_shape = desc.GetShape();
    /// if tensor is a scaler, it's shape size if zero, according ge_tensor.cc.
    /// the logic of GetShapeSize is wrong, the scaler tensor's GetShapeSize is zero
    /// and that of unknown shape is zero too.
    /// unknown shape will not appear here, so we can use zero judge a tensor is scalar or not
    int64_t elem_num = tensor_shape.GetShapeSize();
    if (elem_num == 0 && tensor_shape.GetDims().empty()) {
      elem_num = 1;
    }

    auto &mutable_tensor = const_cast<GeTensor &>(tensor);
    uint64_t *buff = reinterpret_cast<uint64_t *>(mutable_tensor.MutableData().data());
    GE_CHECK_NOTNULL(buff);
    GE_CHK_BOOL_RET_STATUS(ge::CheckInt64Uint32MulOverflow(elem_num, kBytes * kStringHeadElems) == SUCCESS, FAILED,
                           "[Invoke][CheckInt64Uint32MulOverflow] failed because Shape size is invalid.");
    auto offset = static_cast<uint64_t>(elem_num * kBytes * kStringHeadElems);
    auto hbm_raw_data_base_addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(var_addr) + offset);
    for (int64_t i = elem_num - 1; i >= 0; --i) {
      buff[i * kStringHeadElems] = hbm_raw_data_base_addr + (buff[i * kStringHeadElems] - buff[0]);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::AssignUninitializedConstantOps() {
  if (GetContext().GetHostExecFlag()) {
    GELOGI("no need to assign when exec on host.");
    return SUCCESS;
  }
  for (auto &it : constant_op_nodes_) {
    const string &var_name = it.first;
    const NodePtr &var_node = it.second;
    auto tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0);
    if (!var_manager_->IsVarExist(var_name, *tensor_desc)) {
      // allocate constant
      GELOGD("[%s] Constant not allocated during graph building. now allocate it.", var_name.c_str());
      GE_CHK_STATUS_RET(var_manager_->AssignVarMem(var_name, *tensor_desc, RT_MEMORY_HBM));
      GE_CHK_STATUS_RET(var_manager_->SetAllocatedGraphId(var_name, runtime_param_.graph_id));
    }
  }

  for (auto &it : hybrid_model_.device_variable_nodes_) {
    const string &var_name = it.first;
    const NodePtr &var_node = it.second;
    auto tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0);
    if (!var_manager_->IsVarExist(var_name, *tensor_desc)) {
      // allocate constant
      GELOGD("[%s] Constant not allocated during graph building. now allocate it.", var_name.c_str());
      GE_CHK_STATUS_RET(var_manager_->AssignVarMem(var_name, *tensor_desc, RT_MEMORY_HBM));
      GE_CHK_STATUS_RET(VarMemAssignUtil::AssignData2Fp32Var(var_node, runtime_param_.session_id))
      GE_CHK_STATUS_RET(var_manager_->SetAllocatedGraphId(var_name, runtime_param_.graph_id));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitConstantOps() {
  for (auto &it : constant_op_nodes_) {
    const string &var_name = it.first;
    const NodePtr &var_node = it.second;
    auto op_desc = var_node->GetOpDesc();
    auto v_weights = ModelUtils::GetWeights(op_desc);
    if (v_weights.empty()) {
      GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Constant op has no weight", var_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    auto *ge_tensor = const_cast<GeTensor *>(v_weights[0].get());

    std::unique_ptr<TensorValue> var_tensor;
    if (GetContext().GetHostExecFlag()) {
      GE_CHECK_NOTNULL(ge_tensor);
      // Address for eigen kernel should be aligned with 16 bytes
      // Tensors return by api GetWeights share data with proto, whose addr is not confirmed to be aligned
      GeTensor aligned_tensor = ge_tensor->Clone();
      GELOGD("Init tensor with host constant %s size = %zu", var_name.c_str(), aligned_tensor.MutableData().GetSize());
      if (aligned_tensor.GetData().size() > 0) {
        if (MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(aligned_tensor.GetAlignedPtr(),
                                                                         aligned_tensor.GetData().size()) == nullptr) {
          GELOGE(MEMALLOC_FAILED, "[Malloc][HostMemory] for an existed GeTensor failed, model_name_:%s.",
                 GetGraphName());
          return MEMALLOC_FAILED;
        }
        var_tensor.reset(new(std::nothrow)TensorValue(aligned_tensor.MutableData().data(),
                                                      aligned_tensor.GetData().size()));
      } else {
        var_tensor.reset(new(std::nothrow)TensorValue(nullptr, 0));
      }
      GE_CHECK_NOTNULL(var_tensor);
    } else {
      GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, var_tensor));
      GELOGD("Init const op tensor. name = %s, size = %ld", var_name.c_str(), var_tensor->GetSize());
      var_tensor->SetName("ConstOp_" + var_name);
      auto v_output_size = var_tensor->GetSize();
      auto v_output_addr = var_tensor->MutableData();
      if (ge_tensor->GetData().size() > 0) {
        GE_CHK_STATUS_RET_NOLOG(HandleDtString(*ge_tensor, v_output_addr));

        GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p]"
               "mem_size[%zu] datasize[%zu]",
               runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr, v_output_size,
               ge_tensor->GetData().size());
        GE_CHK_RT_RET(rtMemcpy(v_output_addr, v_output_size, ge_tensor->GetData().data(), ge_tensor->GetData().size(),
                               RT_MEMCPY_HOST_TO_DEVICE));
      } else {
        GELOGI("[%s] Const op has no weight data.", op_desc->GetName().c_str());
      }
    }

    hybrid_model_.variable_tensors_.emplace(var_name, std::move(var_tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitVariableTensors() {
  for (auto &it : hybrid_model_.device_variable_nodes_) {
    string var_name = it.first;
    NodePtr &var_node = it.second;
    std::unique_ptr<TensorValue> tensor;
    GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, tensor));
    GELOGD("Init variable tensor. name = %s, size = %ld, addr = %p",
           var_name.c_str(),
           tensor->GetSize(),
           tensor->GetData());
    tensor->SetName("Var_" + var_name);
    hybrid_model_.variable_tensors_.emplace(var_name, std::move(tensor));
  }

  for (const auto &it : hybrid_model_.host_variable_nodes_) {
    auto op_desc = it.second->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(0);
    int64_t tensor_size = 0;
    if (TensorUtils::CalcTensorMemSize(output_tensor.GetShape(), output_tensor.GetFormat(),
                                       output_tensor.GetDataType(), tensor_size) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "CalcTensorMemSize failed, node name:%s", it.first.c_str());
      GELOGE(INTERNAL_ERROR, "[Calculate][TensorMemSize] failed, node name:%s", it.first.c_str());
      return INTERNAL_ERROR;
    }

    // Host variable will be assigned to allocated shared memory first.
    SharedMemInfo mem_info;
    void *mem_addr = nullptr;
    if (HostMemManager::Instance().QueryVarMemInfo(it.first, mem_info)) {
      mem_addr = const_cast<void *>(MemManager::Instance().HostMemInstance(RT_MEMORY_HBM)
                                      .Malloc(mem_info.host_aligned_ptr, tensor_size));
    } else {
      mem_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(tensor_size);
    }

    if (mem_addr == nullptr) {
      REPORT_INNER_ERROR("E19999", "[Malloc][HostMem] for variable [%s] failed.", it.first.c_str());
      GELOGE(MEMALLOC_FAILED, "[Malloc][HostMem] for variable [%s] failed.", it.first.c_str());
      return MEMALLOC_FAILED;
    }
    GELOGD("Host variable [%s] malloc success, size=%ld.", it.first.c_str(), tensor_size);

    std::unique_ptr<TensorValue> tensor(new (std::nothrow) TensorValue(mem_addr, tensor_size));
    GE_CHECK_NOTNULL(tensor);
    hybrid_model_.variable_tensors_.emplace(it.first, std::move(tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitWeights() {
  // For constant in root graph
  for (const auto &subgraph_model : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    const auto &weight_buffer = subgraph_model.second->GetWeight();
    if (weight_buffer.GetSize() == 0) {
      GELOGD("weight is empty");
      return SUCCESS;
    }

    auto allocator = NpuMemoryAllocator::GetAllocator();
    GE_CHECK_NOTNULL(allocator);
    auto sub_weight_buffer = TensorBuffer::Create(allocator, weight_buffer.size());
    GE_CHECK_NOTNULL(sub_weight_buffer);
    auto weight_base = reinterpret_cast<uint8_t *>(sub_weight_buffer->GetData());
    GE_CHK_RT_RET(rtMemcpy(weight_base,
                           sub_weight_buffer->GetSize(),
                           weight_buffer.GetData(),
                           weight_buffer.GetSize(),
                           RT_MEMCPY_HOST_TO_DEVICE));

    GELOGI("Init weight mem successfully, weight base %p, weight size = %zu",
           weight_base,
           sub_weight_buffer->GetSize());
    auto subgraph = GraphUtils::GetComputeGraph(subgraph_model.second->GetGraph());
    if (subgraph != ge_root_model_->GetRootGraph()) {
      subgraph = hybrid_model_.root_graph_->GetSubgraph(subgraph_model.first);
    } else {
      subgraph = hybrid_model_.root_graph_;
    }
    GE_CHECK_NOTNULL(subgraph);
    hybrid_model_.weight_buffer_map_.emplace(subgraph->GetName(), std::move(sub_weight_buffer));
    for (auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() != CONSTANT) {
        continue;
      }

      auto op_desc = node->GetOpDesc();
      auto v_weights = ModelUtils::GetWeights(op_desc);
      if (v_weights.empty()) {
        GELOGE(INTERNAL_ERROR, "[Invoke][GetWeights][%s] Constant has no value", node->GetName().c_str());
        REPORT_CALL_ERROR("E19999", "[%s] Constant has no value.", node->GetName().c_str());
        return INTERNAL_ERROR;
      }
      auto *ge_tensor = const_cast<GeTensor *>(v_weights[0].get());
      GE_CHECK_NOTNULL(ge_tensor);
      const GeTensorDesc &tensor_desc = ge_tensor->GetTensorDesc();
      int64_t tensor_size = 0;
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetSize(*op_desc->MutableOutputDesc(0), tensor_size),
                              "[Invoke][GetSize][%s] Failed to get output tensor size",
                              node->GetName().c_str());
      int64_t data_offset = 0;
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetDataOffset(tensor_desc, data_offset),
                              "[Invoke][GetDataOffset][%s] Failed to get data offset",
                              node->GetName().c_str());
      GELOGD("[%s] Start to init Constant node [%s], size = %ld, offset = %ld",
             GetGraphName(),
             node->GetName().c_str(),
             tensor_size,
             data_offset);

      auto tensor_buffer = TensorBuffer::Create(weight_base + data_offset, tensor_size);
      GE_CHECK_NOTNULL(tensor_buffer);
      std::unique_ptr<TensorValue> constant_tensor(new (std::nothrow)TensorValue(std::move(tensor_buffer)));
      GE_CHECK_NOTNULL(constant_tensor);
      constant_tensor->SetName("Constant_" + op_desc->GetName());
      hybrid_model_.constant_tensors_.emplace(node, std::move(constant_tensor));
      GELOGD("[%s] Constant node [%s] added, size = %ld", GetGraphName(), node->GetName().c_str(), tensor_size);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::LoadTask(NodeItem &node_item) {
  auto &node_ptr = node_item.node;
  GELOGD("[%s] Start to build kernel task", node_ptr->GetName().c_str());
  auto load_ret = node_item.node_executor->LoadTask(hybrid_model_,
                                                    node_ptr,
                                                    node_item.kernel_task);
  if (load_ret != UNSUPPORTED && load_ret != SUCCESS) {
    GELOGE(load_ret, "[Invoke][LoadTask][%s] Failed to load task", node_ptr->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "[%s] Failed to load task", node_ptr->GetName().c_str());
    return load_ret;
  }

  GELOGD("[%s] Done loading task successfully.", node_ptr->GetName().c_str());
  return SUCCESS;
}

Status HybridModelBuilder::LoadTasks() {
  GE_CHK_STATUS_RET(CheckAicpuOpList(), "[Check][AicpuOpList] failed.");
  std::map<int, std::map<std::string, NodeItem *>> ordered_partitioned_calls;
  for (auto &it : hybrid_model_.node_items_) {
    auto &node_item = it.second;
    if (node_item->node_type == NETOUTPUT) {
      continue;
    }
    if (node_item->node_type == PARTITIONEDCALL) {
      ordered_partitioned_calls[node_item->node_id][node_item->node_name] = node_item.get();
      continue;
    }
    GE_CHK_STATUS_RET_NOLOG(LoadTask(*node_item));
  }

  // HCCL operators need to be loaded in the same order across different processes
  for (auto &it : ordered_partitioned_calls) {
    for (auto &it2 : it.second) {
      GE_CHK_STATUS_RET_NOLOG(LoadTask(*it2.second));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGeModel(ComputeGraph &sub_graph, const GeModelPtr &ge_model) {
  auto parent_node = sub_graph.GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  auto op_type = parent_node->GetType();
  if (IsControlFlowV2Op(op_type)) {
    GELOGD("Set ge_model for control op subgraph: [%s], task_size = %d",
           sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    subgraph_models_.emplace(sub_graph.GetName(), ge_model);
  } else {
    GELOGD("Set ge_model for subgraph: [%s], task_size = %d",
           sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    hybrid_model_.known_shape_sub_models_.emplace(parent_node, ge_model);
  }

  GE_CHK_STATUS_RET_NOLOG(InitHcclExecutorOnDemand(ge_model));
  return SUCCESS;
}

Status HybridModelBuilder::InitHcclExecutorOnDemand(const GeModelPtr &ge_model) {
  if (NodeExecutorManager::GetInstance().IsExecutorInitialized(NodeExecutorManager::ExecutorType::HCCL)) {
    return SUCCESS;
  }

  // HCCL tasks in known-shaped subgraph which resides in a dynamic root graph
  // still depends on the initialization of the HcclExecutor
  auto tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    if (task_type == RT_MODEL_TASK_HCCL) {
      const NodeExecutor *unused = nullptr;
      GE_CHK_STATUS_RET_NOLOG(NodeExecutorManager::GetInstance()
                                  .GetOrCreateExecutor(NodeExecutorManager::ExecutorType::HCCL, &unused));
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::IndexTaskDefs(const ComputeGraphPtr &sub_graph, const GeModelPtr &ge_model) {
  // index task defs
  GELOGD("To index tasks for subgraph: %s", sub_graph->GetName().c_str());
  std::unordered_map<int64_t, NodePtr> node_map;
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto node_id = node->GetOpDesc()->GetId();
    GELOGD("op_index = %ld, node_name = %s", node_id, node->GetName().c_str());
    node_map.emplace(node_id, node);
  }

  auto tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    GELOGI("Task id = %d, task type = %d", i, task_def.type());
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    uint32_t op_index = -1;
    if (task_type == RT_MODEL_TASK_KERNEL) {
      op_index = task_def.kernel().context().op_index();
    } else if (task_type == RT_MODEL_TASK_KERNEL_EX) {
      op_index = task_def.kernel_ex().op_index();
    } else if (task_type == RT_MODEL_TASK_HCCL) {
      op_index = task_def.kernel_hccl().op_index();
    } else if (task_type == RT_MODEL_TASK_ALL_KERNEL) {
      op_index = task_def.kernel_with_handle().context().op_index();
    } else {
      GELOGD("Skip task type: %d", static_cast<int>(task_type));
      continue;
    }
    GELOGD("op_index = %u, task_type = %d", op_index, task_type);

    auto iter = node_map.find(op_index);
    if (iter == node_map.end()) {
      GELOGE(INTERNAL_ERROR, "[Find][Node]Failed to get node by op_index = %u", op_index);
      REPORT_INNER_ERROR("E19999", "Failed to get node by op_index = %u.", op_index);
      return INTERNAL_ERROR;
    }

    auto &node = iter->second;
    if (task_type == RT_MODEL_TASK_KERNEL || task_type == RT_MODEL_TASK_ALL_KERNEL) {
      ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(node->GetOpDesc());
    }

    GELOGD("Task loaded for node: %s, task type = %d, op_index = %u", node->GetName().c_str(), task_type, op_index);
    hybrid_model_.task_defs_[node].emplace_back(task_def);
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexTaskDefs() {
  const auto &root_graph = hybrid_model_.root_graph_;
  const auto &root_graph_name = root_graph->GetName();
  if (SetOutputNameAttr(*root_graph) != SUCCESS) {
    GELOGW("Set output name attr failed.");
  }

  for (auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    auto &name = it.first;
    auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);

    auto sub_graph = root_graph->GetSubgraph(name);
    if (name != root_graph_name) {
      if (sub_graph == nullptr) {
        continue;
      }

      bool is_unknown_shape = sub_graph->GetGraphUnknownFlag();
      if (!is_unknown_shape) {
        GE_CHK_STATUS_RET_NOLOG(LoadGeModel(*sub_graph, ge_model));
        continue;
      }
    } else {
      sub_graph = root_graph;
    }

    GE_CHK_STATUS_RET_NOLOG(IndexTaskDefs(sub_graph, ge_model));
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexSpecialNodes() {
  GELOGD("Start to index special nodes");
  const auto &root_graph = hybrid_model_.root_graph_;
  for (auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto op_type = node->GetType();
    GELOGD("node name = %s, node type = %s", node->GetName().c_str(), node->GetType().c_str());
    if (op_type == VARIABLE) {
      string placement;
      (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_VARIABLE_PLACEMENT, placement);
      if (placement == "host") {
        hybrid_model_.host_variable_nodes_.emplace(node->GetName(), node);
      } else {
        hybrid_model_.device_variable_nodes_.emplace(node->GetName(), node);
      }
    } else if (op_type == CONSTANTOP) {
      constant_op_nodes_.emplace(node->GetName(), node);
    } else if (op_type == STREAMMERGE) {
      stream_merge_op_nodes_.emplace(node->GetName(), node);
    } else if (op_type == NEXTITERATION || op_type == REFNEXTITERATION) {
      next_iteration_op_nodes_.emplace(node->GetName(), node);
    } else if (op_type == DATA && node->GetOwnerComputeGraph() != root_graph) {
      NodePtr src_node;
      int peer_out_index = -1;
      GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_out_index));
      GELOGD("Got peer node for data node %s, peer node = %s(%s)",
             node->GetName().c_str(),
             src_node->GetName().c_str(),
             src_node->GetType().c_str());

      auto src_op_type = src_node->GetType();
      if (src_op_type == CONSTANTOP || src_op_type == VARIABLE) {
        for (auto &dst_node_and_in_anchor : node->GetOutDataNodesAndAnchors()) {
          auto &dst_node = dst_node_and_in_anchor.first;
          auto &in_anchor = dst_node_and_in_anchor.second;
          node_ref_inputs_[dst_node].emplace_back(std::make_pair(in_anchor->GetIdx(), src_node));
        }
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::GetPeerNodeAcrossSubGraphs(const NodePtr &data_node,
                                                      NodePtr &peer_node,
                                                      int &peer_out_index) {
  auto sub_graph = data_node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(sub_graph);
  GELOGD("To get peer node of %s::%s", sub_graph->GetName().c_str(), data_node->GetName().c_str());
  auto wrapped_node = data_node->GetOwnerComputeGraph()->GetParentNode();
  if (wrapped_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "[%s] Node is in root graph.", data_node->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][GetParentNode][%s] Node is in root graph.", data_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto data_op_desc = data_node->GetOpDesc();
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    REPORT_CALL_ERROR("E19999", "[%s] Failed to get attr [%s].", data_op_desc->GetName().c_str(),
                      ATTR_NAME_PARENT_NODE_INDEX.c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][GetInt][%s] Failed to get attr [%s]",
           data_op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return INTERNAL_ERROR;
  }

  auto wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL(wrapped_node_in_anchor);
  auto src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
    REPORT_INNER_ERROR("E19999", "[%s] Parent node do not have peer anchor.", data_node->GetName().c_str());
    GELOGE(INTERNAL_ERROR,
           "[Check][ParentNode][%s] Parent node do not have peer anchor.", data_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  auto src_wrapped_node_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_wrapped_node_out_anchor);
  auto src_wrapped_node = src_wrapped_node_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_wrapped_node);

  // connected to root-graph's DATA
  auto src_node_type = src_wrapped_node->GetType();
  if (src_node_type != PARTITIONEDCALL) {
    peer_node = src_wrapped_node;
    peer_out_index = kVarOutputIndex;
    GELOGD("[%s] Node is connected to root graph's node: %s",
           data_node->GetName().c_str(),
           peer_node->GetName().c_str());
    return SUCCESS;
  }

  auto src_graph = NodeUtils::GetSubgraph(*src_wrapped_node, kSubgraphIndex);
  GE_CHECK_NOTNULL(src_graph);
  auto src_net_output_node = src_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (src_net_output_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to find NetOutput in subgraph: %s", src_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][FindFirstNodeMatchType]Failed to find NetOutput in subgraph: %s",
           src_graph->GetName().c_str());
    return INTERNAL_ERROR;
  }
  auto net_output_desc = src_net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  auto out_index = static_cast<uint32_t>(src_wrapped_node_out_anchor->GetIdx());
  GELOGD("src graph = %s, src parent output index = %u", src_graph->GetName().c_str(), out_index);

  // link src to outputs of DataNode
  auto input_size = net_output_desc->GetAllInputsSize();
  GE_CHECK_LE(input_size, UINT32_MAX);
  for (uint32_t i = 0; i < static_cast<uint32_t>(input_size); ++i) {
    uint32_t p_index = 0;
    if (!AttrUtils::GetInt(net_output_desc->GetInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
      GELOGW("SubGraph: %s input tensor %u attr %s not found.",
             src_graph->GetName().c_str(), i, ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    GELOGD("NetOutput's input[%u], parent_node_index = %u", i, p_index);
    if (p_index == out_index) {
      auto in_anchor = src_net_output_node->GetInDataAnchor(i);
      GE_CHECK_NOTNULL(in_anchor);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      peer_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      peer_out_index = peer_out_anchor->GetIdx();
      GELOGD("Found peer node of Data node: %s::%s is %s::%s",
             sub_graph->GetName().c_str(),
             data_node->GetName().c_str(),
             src_graph->GetName().c_str(),
             peer_node->GetName().c_str());
      return SUCCESS;
    }
  }

  GELOGE(FAILED, "[Get][PeerNode]Failed to find peer node for %s::%s", sub_graph->GetName().c_str(),
         data_node->GetName().c_str());
  REPORT_INNER_ERROR("E19999", "Failed to find peer node for %s::%s.",
                     sub_graph->GetName().c_str(), data_node->GetName().c_str());
  return FAILED;
}
Status HybridModelBuilder::InitRuntimeParams() {
  int64_t value = 0;
  bool ret = false;
  if (ge_root_model_->GetSubgraphInstanceNameToModel().empty()) {
    GELOGE(INTERNAL_ERROR, "[Get][SubModel]Root model has no sub model, model:%s.", GetGraphName());
    REPORT_INNER_ERROR("E19999", "Root model has no sub model, model:%s.", GetGraphName());
    return INTERNAL_ERROR;
  }

  // session id and var size is same for every model
  auto first_model = ge_root_model_->GetSubgraphInstanceNameToModel().begin()->second;
  ret = ge::AttrUtils::GetInt(first_model, ge::MODEL_ATTR_SESSION_ID, value);
  runtime_param_.session_id = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(first_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? static_cast<uint64_t>(value) : 0;
  runtime_param_.graph_id = hybrid_model_.root_graph_->GetGraphID();
  value = 0;
  for (auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    (void) ge::AttrUtils::GetInt(it.second, ATTR_MODEL_VAR_SIZE, value);
    if (value > 0) {
      runtime_param_.var_size = static_cast<uint64_t>(value);
      break;
    }
  }

  GELOGI("InitRuntimeParams(), session_id:%lu, var_size:%lu. graph_id = %u",
         runtime_param_.session_id, runtime_param_.var_size, runtime_param_.graph_id);

  var_manager_ = VarManager::Instance(runtime_param_.session_id);
  GE_CHECK_NOTNULL(var_manager_);
  return SUCCESS;
}

Status HybridModelBuilder::IdentifyVariableOutputs(NodeItem &node_item, const ComputeGraphPtr &subgraph) {
  GELOGD("Start to parse outputs of node: %s", node_item.NodeName().c_str());
  auto net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("[%s] Subgraph do not got net output", subgraph->GetName().c_str());
    return SUCCESS;
  }
  auto net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  // constants connected to net output
  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    auto src_node = GetPeerNode(in_data_anchor);
    GE_CHECK_NOTNULL(src_node);
    auto src_op_type = src_node->GetType();
    if (src_op_type == CONSTANTOP || src_op_type == CONSTANT) {
      known_subgraph_constant_output_refs_[&node_item].emplace(in_data_anchor->GetIdx(), src_node);
    }
  }

  // Data nodes marked with REF_VAR_SRC_VAR_NAME
  // Using variable tensor as data's output
  for (auto &node : subgraph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }

    string ref_var_name;
    (void) AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_name);
    if (ref_var_name.empty()) {
      continue;
    }

    GELOGD("Data node ref to variable: %s", ref_var_name.c_str());
    NodePtr src_node;
    auto var_node = hybrid_model_.GetVariableNode(ref_var_name);
    GE_CHECK_NOTNULL(var_node);
    GELOGD("Found var node [%s] by ref_var_name [%s]", var_node->GetName().c_str(), ref_var_name.c_str());
    int peer_output_index = -1;
    GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_output_index));
    auto src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->ref_outputs.emplace(peer_output_index, var_node);
  }

  return SUCCESS;
}

NodePtr HybridModelBuilder::GetPeerNode(const InDataAnchorPtr &in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    return peer_out_anchor->GetOwnerNode();
  }

  return nullptr;
}

Status HybridModelBuilder::GetParentNodeOutputIndex(const OpDesc &op_desc, int index, uint32_t &out_index) {
  auto input_desc = op_desc.MutableInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, out_index)) {
    GELOGE(INTERNAL_ERROR, "[Invoke][GetInt]NetOutput %s input tensor %d, attr %s not found.",
           op_desc.GetName().c_str(), index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
    REPORT_CALL_ERROR("E19999", "NetOutput %s input tensor %d, attr %s not found.",
                      op_desc.GetName().c_str(), index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status HybridModelBuilder::InitModelMem() {
  hybrid_model_.var_mem_base_ = var_manager_->GetVarMemoryBase(RT_MEMORY_HBM);
  auto total_var_size = hybrid_model_.TotalVarMemSize();
  if (total_var_size == 0 && !constant_op_nodes_.empty()) {
    total_var_size = var_manager_->GetVarMemSize(RT_MEMORY_HBM) > 0 ? var_manager_->GetVarMemMaxSize() : 0;
    GELOGD("Model var size = 0. but got uninitialized constant. set var size to %zu.", total_var_size);
  }

  if (total_var_size > 0 && hybrid_model_.var_mem_base_ == nullptr) {
    GE_CHK_STATUS_RET(var_manager_->MallocVarMemory(total_var_size),
                      "[Malloc][VarMemory] failed, size:%zu.", total_var_size);
    hybrid_model_.var_mem_base_ = var_manager_->GetVarMemoryBase(RT_MEMORY_HBM);
  }

  runtime_param_.var_base = hybrid_model_.var_mem_base_;
  auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  hybrid_model_.global_step_ = TensorBuffer::Create(allocator, sizeof(int64_t));
  GE_CHECK_NOTNULL(hybrid_model_.global_step_);
  return SUCCESS;
}

Status HybridModelBuilder::TransAllVarData() {
  GELOGI("TransAllVarData start: session_id:%lu, graph_id: %u.", runtime_param_.session_id, runtime_param_.graph_id);
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Invoke][rtCtxGetCurrent]Failed to get current context, error_code is: 0x%X.", rt_ret);
    REPORT_CALL_ERROR("E19999", "rtCtxGetCurrent failed, error_code: 0x%X.", rt_ret);
    return RT_FAILED;
  }

  std::vector<NodePtr> variable_node_list;
  for (auto &it : hybrid_model_.device_variable_nodes_) {
    variable_node_list.emplace_back(it.second);
    GELOGD("[%s] added for trans var data", it.first.c_str());
  }

  GE_CHK_STATUS_RET(TransVarDataUtils::TransAllVarData(variable_node_list,
                                                       runtime_param_.session_id,
                                                       ctx,
                                                       runtime_param_.graph_id),
                    "[Invoke][TransAllVarData] failed.");

  GELOGI("TransAllVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::CopyVarData() {
  GE_CHK_STATUS_RET(TransVarDataUtils::CopyVarData(hybrid_model_.root_graph_,
                                                   runtime_param_.session_id,
                                                   hybrid_model_.device_id_),
                    "[Invoke][CopyVarData] failed.");
  GELOGI("CopyVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::LoadKnownShapedSubgraph(ComputeGraph &graph, NodeItem *parent_node_item) {
  GELOGD("Start to load known shaped subgraph [%s]", graph.GetName().c_str());
  auto graph_item = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  GE_CHECK_NOTNULL(graph_item);
  graph_item->is_dynamic_ = false;
  auto subgraph_name = graph.GetName();
  auto wrapper_op_desc = MakeShared<OpDesc>(subgraph_name + "_partitioned_call", PARTITIONEDCALL);
  GE_CHECK_NOTNULL(wrapper_op_desc);

  for (auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &op_type = node->GetType();

    if (op_type == DATA) {
      int32_t data_index = 0;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGE(FAILED,
               "[Invoke][GetInt][%s] Failed to get attr [%s]",
               node->GetName().c_str(),
               ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }

      (void) wrapper_op_desc->AddInputDesc(op_desc->GetInputDesc(0));
      graph_item->input_index_mapping_.emplace_back(data_index);
    } else if (op_type == NETOUTPUT) {
      int output_index = 0;
      for (const auto &output_desc : op_desc->GetAllInputsDescPtr()) {
        int32_t data_index = output_index++;
        if (!AttrUtils::GetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
          GELOGI("[%s] Failed to get attr [%s]", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        }

        GE_CHK_GRAPH_STATUS_RET(wrapper_op_desc->AddOutputDesc(*output_desc),
                                "[Invoke][AddOutputDesc][%s] Failed to add output desc. output index = %d",
                                graph.GetName().c_str(),
                                output_index);

        graph_item->output_index_mapping_.emplace_back(data_index);
      }
    }
  }

  auto temp_graph = MakeShared<ComputeGraph>("temp");
  GE_CHECK_NOTNULL(temp_graph);
  auto wrapper_node = temp_graph->AddNode(wrapper_op_desc);
  wrapper_op_desc->SetId(parent_node_item->node_id);
  GeModelPtr ge_model = subgraph_models_[subgraph_name];
  GE_CHECK_NOTNULL(ge_model);
  hybrid_model_.known_shape_sub_models_.emplace(wrapper_node, ge_model);

  NodeItem *node_item = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(wrapper_node, &node_item));
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->outputs.resize(node_item->num_outputs);
  graph_item->node_items_.emplace_back(node_item);
  graph_item->output_node_ = node_item;
  graph_item->total_inputs_ = node_item->num_inputs;
  graph_item->total_outputs_ = node_item->num_outputs;

  GELOGD("NodeItem create for known shape subgraph [%s], NodeItem = %s",
         graph.GetName().c_str(),
         node_item->DebugString().c_str());

  GELOGD("Done parse known shape subgraph successfully. graph = [%s]", graph.GetName().c_str());
  graph_item->SetName(graph.GetName());
  GELOGD("Done loading known shape subgraph: [%s]", graph_item->GetName().c_str());
  hybrid_model_.subgraph_items_.emplace(graph.GetName(), std::move(graph_item));
  return SUCCESS;
}

Status HybridModelBuilder::RecoverGraphUnknownFlag() {
  const auto &root_graph = hybrid_model_.root_graph_;
  for (auto &sub_graph : root_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    for (const auto &node : sub_graph->GetDirectNode()) {
      bool is_unknown_shape = false;
      (void)AttrUtils::GetBool(node->GetOpDesc(), kOwnerGraphIsUnknown, is_unknown_shape);
      sub_graph->SetGraphUnknownFlag(is_unknown_shape);
      break;
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::GenerateFpProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list) {
  uint64_t jobid_log_id = ge::GetContext().TraceId();
  GELOGD("The first FP operator is %s,, job_id %lu", op_desc->GetName().c_str(), jobid_log_id);

  TaskDef job_task_def;
  job_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
  job_task_def.set_stream_id(op_desc->GetStreamId());
  LogTimeStampDef *job_log_def = job_task_def.mutable_log_timestamp();
  if (job_log_def != nullptr) {
    job_log_def->set_logid(jobid_log_id);
    job_log_def->set_notify(false);
  }
  task_def_list.emplace_back(job_task_def);
  TaskDef fp_task_def;
  fp_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
  fp_task_def.set_stream_id(op_desc->GetStreamId());
  LogTimeStampDef *fp_log_def = fp_task_def.mutable_log_timestamp();
  if (fp_log_def != nullptr) {
    fp_log_def->set_logid(kProfilingFpStartLogid);
    fp_log_def->set_notify(false);
  }
  task_def_list.emplace_back(fp_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateArProfilingTask(const OpDescPtr &op_desc, int64_t log_id,
                                                   vector<domi::TaskDef> &task_def_list) {
  TaskDef ar_task_def;
  ar_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
  ar_task_def.set_stream_id(op_desc->GetStreamId());
  LogTimeStampDef *ar_log_def = ar_task_def.mutable_log_timestamp();
  if (ar_log_def != nullptr) {
    ar_log_def->set_logid(log_id);
    ar_log_def->set_notify(false);
  }
  task_def_list.emplace_back(ar_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateBpProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list) {
    TaskDef bp_task_def;
    bp_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
    bp_task_def.set_stream_id(op_desc->GetStreamId());
    LogTimeStampDef *bp_log_def = bp_task_def.mutable_log_timestamp();
    GE_CHECK_NOTNULL(bp_log_def);
    bp_log_def->set_logid(kProfilingBpEndLogid);
    bp_log_def->set_notify(false);
    task_def_list.emplace_back(bp_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateEndProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list) {
  TaskDef end_task_def;
  end_task_def.set_type(RT_MODEL_TASK_PROFILER_TRACE);
  end_task_def.set_stream_id(op_desc->GetStreamId());
  LogTimeStampDef *end_log_def = end_task_def.mutable_log_timestamp();
  GE_CHECK_NOTNULL(end_log_def);
  end_log_def->set_logid(kProfilingIterEndLogid);
  end_log_def->set_notify(true);
  task_def_list.emplace_back(end_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::CreateProfilingNodeBefore(GraphItem &graph_item, const NodePtr &node, uint32_t &prev_num) {
  GE_CHECK_NOTNULL(node);
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &compute_graph = MakeShared<ComputeGraph>(kProfilingGraph);
  GE_CHECK_NOTNULL(compute_graph);

  NodePtr node_ptr = nullptr;
  map<NodePtr, vector<domi::TaskDef>> node_task_map;
  // create fp node
  bool is_insert_fp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_FP_PROFILILNG_TASK, is_insert_fp_profiling_task);
  if (is_insert_fp_profiling_task) {
    vector<domi::TaskDef> task_def_list;
    (void)GenerateFpProfilingTask(op_desc, task_def_list);
    auto fp_desc = MakeShared<OpDesc>(kProfilingFpNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(fp_desc);
    fp_desc->SetOpKernelLibName(kEngineNameRts);
    node_ptr = compute_graph->AddNode(fp_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create fp profiling node success before.");
  }
  // creat all reduce start node
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if (is_all_reduce && is_insert_bp_profiling_task) {
    vector<domi::TaskDef> task_def_list;
    int64_t log_id = 0;
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, log_id);
    GELOGD("All reduce node profiling task log id: %ld before", log_id);
    (void) GenerateArProfilingTask(op_desc, log_id, task_def_list);
    string op_name = string(kProfilingArNode) + std::to_string(log_id);
    auto ar_desc_start = MakeShared<OpDesc>(op_name, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(ar_desc_start);
    ar_desc_start->SetOpKernelLibName(kEngineNameRts);
    node_ptr = compute_graph->AddNode(ar_desc_start);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create all reduce start profiling node success before.");
  }

  if (!node_task_map.empty()) {
    for (const auto &node_task : node_task_map) {
      NodePtr profiling_node = node_task.first;
      const vector<domi::TaskDef> &task_def_lists = node_task.second;
      for (const auto &task_def : task_def_lists) {
        hybrid_model_.task_defs_[profiling_node].emplace_back(task_def);
      }
      if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
        uint32_t stage_level = UINT32_MAX;
        (void)ge::AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, stage_level);
        (void)ge::AttrUtils::SetInt(node_ptr->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
      }
      NodeItem *node_item = nullptr;
      GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(profiling_node, &node_item));
      GE_CHECK_NOTNULL(node_item);
      node_item->input_start = 0;
      node_item->output_start = 0;
      graph_item.node_items_.emplace_back(node_item);
      ++prev_num;
    }
  } else {
    GELOGD("No need to create profiling node before.");
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateProfilingNodeAfter(GraphItem &graph_item, const NodePtr &node, uint32_t &post_num) {
  GE_CHECK_NOTNULL(node);
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &compute_graph = MakeShared<ComputeGraph>(kProfilingGraph);
  GE_CHECK_NOTNULL(compute_graph);

  NodePtr node_ptr = nullptr;
  map<NodePtr, vector<domi::TaskDef>> node_task_map;
  // Create all reduce end node
  bool is_insert_bp_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  bool is_all_reduce = (op_desc->GetType() == HCOMALLREDUCE || op_desc->GetType() == HVDCALLBACKALLREDUCE);
  if (is_all_reduce && is_insert_bp_profiling_task) {
    vector<domi::TaskDef> task_def_list;
    int64_t log_id = 0;
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, log_id);
    GELOGD("All reduce node profiling task log id: %ld after", log_id);
    (void) GenerateArProfilingTask(op_desc, log_id + 1, task_def_list);
    string op_name = string(kProfilingArNode) + std::to_string(log_id + 1);
    auto ar_desc_end = MakeShared<OpDesc>(op_name, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(ar_desc_end);
    ar_desc_end->SetOpKernelLibName(kEngineNameRts);
    node_ptr = compute_graph->AddNode(ar_desc_end);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create all reduce end profiling node success after.");
  }
  // create bp node
  if (!is_all_reduce && is_insert_bp_profiling_task) {
    vector<domi::TaskDef> task_def_list;
    (void) GenerateBpProfilingTask(op_desc, task_def_list);
    auto bp_op_desc = MakeShared<OpDesc>(kProfilingBpNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(bp_op_desc);
    bp_op_desc->SetOpKernelLibName(kEngineNameRts);
    node_ptr = compute_graph->AddNode(bp_op_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create bp profiling node success after.");
  }
  // create end node
  bool is_insert_end_profiling_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_END_PROFILILNG_TASK, is_insert_end_profiling_task);
  if (is_insert_end_profiling_task) {
    vector<domi::TaskDef> task_def_list;
    (void)GenerateEndProfilingTask(op_desc, task_def_list);
    auto end_desc = MakeShared<OpDesc>(kProfilingEndNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(end_desc);
    end_desc->SetOpKernelLibName(kEngineNameRts);
    node_ptr = compute_graph->AddNode(end_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create end profiling node success after.");
  }

  if (!node_task_map.empty()) {
    for (const auto &node_task : node_task_map) {
      NodePtr profiling_node = node_task.first;
      const vector<domi::TaskDef> &task_def_lists = node_task.second;
      for (const auto &task_def : task_def_lists) {
        hybrid_model_.task_defs_[profiling_node].emplace_back(task_def);
      }
      if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
        uint32_t stage_level = UINT32_MAX;
        (void)ge::AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, stage_level);
        (void)ge::AttrUtils::SetInt(profiling_node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
      }
      NodeItem *node_item = nullptr;
      GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(profiling_node, &node_item));
      GE_CHECK_NOTNULL(node_item);
      node_item->input_start = 0;
      node_item->output_start = 0;
      graph_item.node_items_.emplace_back(node_item);
      ++post_num;
    }
  } else {
    GELOGD("No need to create profiling node after.");
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadDynamicSubgraph(ComputeGraph &graph, bool is_root_graph) {
  GELOGD("Start to load subgraph [%s]", graph.GetName().c_str());
  // for known partitioned call, load all nodes
  auto graph_item = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  GE_CHECK_NOTNULL(graph_item);

  graph_item->is_dynamic_ = true;
  graph_item->node_items_.reserve(graph.GetDirectNodesSize());
  int input_start = 0;
  int output_start = 0;
  std::vector<NodeItem *> data_nodes;
  std::map<size_t, std::pair<uint32_t, uint32_t>> profiling_nodes;
  for (auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto &op_type = node->GetType();

    NodeItem *node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, &node_item));
    GE_CHK_STATUS_RET_NOLOG(BuildNodeItem(node, *node_item));
    GE_CHK_STATUS_RET_NOLOG(UpdateAnchorStatus(node)); // needed by FE generate task

    GE_CHK_STATUS_RET_NOLOG(BuildFrameGroupIndex(*node_item));
    GE_CHK_STATUS_RET_NOLOG(BuildControlFlowGroup(*graph_item, node, node_item));
    if (node->GetInAllNodes().empty()) {
      graph_item->root_items_.emplace_back(node_item);
      GELOGD("[%s] add to root node list", node->GetName().c_str());
    }

    node_item->input_start = input_start;
    node_item->output_start = output_start;
    input_start += node_item->num_inputs;
    output_start += node_item->num_outputs;

    if (op_type == DATA_TYPE || op_type == AIPP_DATA_TYPE) {
      data_nodes.emplace_back(node_item);
    } else if (op_type == NETOUTPUT) {
      graph_item->output_node_ = node_item;
      GE_CHK_STATUS_RET_NOLOG(BuildOutputMapping(*graph_item, *node_item, is_root_graph));
    }

    uint32_t prev_num = 0;
    uint32_t post_num = 0;
    GE_CHK_STATUS_RET_NOLOG(CreateProfilingNodeBefore(*graph_item, node, prev_num));
    size_t node_index = graph_item->node_items_.size();
    graph_item->node_items_.emplace_back(node_item);
    GE_CHK_STATUS_RET_NOLOG(CreateProfilingNodeAfter(*graph_item, node, post_num));
    if (prev_num > 0 || post_num > 0) {
      profiling_nodes[node_index] = { prev_num, post_num };
    }
    // parse var outputs
    GE_CHK_STATUS_RET_NOLOG(ParseVarOutputs(*node_item));
    GELOGD("NodeItem created: %s", node_item->DebugString().c_str());
  }

  graph_item->total_inputs_ = input_start;
  graph_item->total_outputs_ = output_start;
  GE_CHK_STATUS_RET_NOLOG(BuildInputMapping(*graph_item, data_nodes, is_root_graph));
  GE_CHK_STATUS_RET_NOLOG(BuildProfilingControl(*graph_item, profiling_nodes));
  if (is_root_graph) {
    graph_item->SetName("Root-Graph");
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    hybrid_model_.root_graph_item_ = std::move(graph_item);
  } else {
    graph_item->SetName(graph.GetName());
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    hybrid_model_.subgraph_items_.emplace(graph.GetName(), std::move(graph_item));
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseVarOutputs(NodeItem &node_item) {
  for (int i = 0; i < node_item.num_outputs; ++i) {
    auto output_tensor_desc = node_item.op_desc->GetOutputDesc(i);
    std::string var_name;
    (void) AttrUtils::GetStr(output_tensor_desc, ASSIGN_VAR_NAME, var_name);
    if (!var_name.empty()) {
      auto var_node = hybrid_model_.GetVariableNode(var_name);
      GE_CHECK_NOTNULL(var_node);
      node_item.ref_outputs.emplace(i, var_node);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::BuildInputMapping(GraphItem &graph_item,
                                             vector<NodeItem *> &data_nodes,
                                             bool is_root_graph) {
  uint32_t data_op_index = 0;
  for (auto &node_item : data_nodes) {
    auto node = node_item->node;
    int data_index = data_op_index;
    if (is_root_graph) {
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, data_index)) {
        GELOGI("ge_train: get new index %u, old %u", data_index, data_op_index);
      }
      data_op_index++;
    } else {
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGE(FAILED, "[Invoke][GetInt][%s] Failed to get attr [%s]",
               node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        REPORT_CALL_ERROR("E19999", "call GetInt failed, [%s] Failed to get attr [%s]",
                          node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }
    }

    if (graph_item.input_nodes_.size() <= static_cast<size_t>(data_index)) {
      graph_item.input_nodes_.resize(data_index + 1);
    }

    graph_item.input_nodes_[data_index] = node_item;
  }

  return SUCCESS;
}

Status HybridModelBuilder::CheckAicpuOpList() {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  std::set<std::string> aicpu_optype_set;
  std::set<std::string> aicpu_tf_optype_set;
  for (auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);
    if (ge::AttrUtils::GetListStr(*ge_model, "needCheckCpu", aicpu_optype_list)) {
      aicpu_optype_set.insert(aicpu_optype_list.begin(), aicpu_optype_list.end());
    }

    if (ge::AttrUtils::GetListStr(*ge_model, "needCheckTf", aicpu_tf_optype_list)) {
      aicpu_tf_optype_set.insert(aicpu_tf_optype_list.begin(), aicpu_tf_optype_list.end());
    }
  }
  // reset list with set
  aicpu_optype_list.assign(aicpu_optype_set.begin(), aicpu_optype_set.end());
  aicpu_tf_optype_list.assign(aicpu_tf_optype_set.begin(), aicpu_tf_optype_set.end());
  GE_CHK_STATUS_RET(ModelManager::GetInstance()->LaunchKernelCheckAicpuOp(aicpu_optype_list, aicpu_tf_optype_list),
                    "[Launch][KernelCheckAicpuOp] failed.");
  return SUCCESS;
}

Status HybridModelBuilder::CollectParallelGroups(NodeItem *node_item) {
  const auto &node = node_item->node;
  auto executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*node);
  if (executor_type == NodeExecutorManager::ExecutorType::HCCL) {
    std::string parallel_group;
    if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, parallel_group)) {
      GELOGD("[%s] Got parallel group = [%s]", node_item->NodeName().c_str(), parallel_group.c_str());
      parallel_group_to_nodes_[parallel_group].emplace(node_item);
      std::set<std::string> group{parallel_group};
      node_to_parallel_groups_[node_item].emplace(parallel_group);
    }
  } else if (executor_type == NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH) {
    std::set<std::string> parallel_groups;
    GELOGD("[%s] To collect parallel group for known-shaped subgraph", node_item->NodeName().c_str());
    for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
      GELOGD("[%s] Start to get parallel group from subgraph: %s",
             node_item->NodeName().c_str(),
             subgraph_name.c_str());
      auto subgraph = hybrid_model_.root_graph_->GetSubgraph(subgraph_name);
      GE_CHECK_NOTNULL(subgraph);
      for (const auto &sub_node : subgraph->GetAllNodes()) {
        std::string parallel_group;
        if (AttrUtils::GetStr(sub_node->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, parallel_group)) {
          GELOGD("[%s::%s] Got parallel group = %s",
                 subgraph_name.c_str(),
                 sub_node->GetName().c_str(),
                 parallel_group.c_str());
          parallel_groups.emplace(parallel_group);
        }
      }
    }

    if (!parallel_groups.empty()) {
      for (const auto &parallel_group : parallel_groups) {
        parallel_group_to_nodes_[parallel_group].emplace(node_item);
        GELOGD("[%s] has parallel group: %s", node_item->NodeName().c_str(), parallel_group.c_str());
      }
      node_to_parallel_groups_.emplace(node_item, std::move(parallel_groups));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentByParallelGroup() {
  for (auto &it : hybrid_model_.node_items_) {
    GE_CHK_STATUS_RET_NOLOG(CollectParallelGroups(it.second.get()));
  }
  for (const auto &it : node_to_parallel_groups_) {
    auto node_item = it.first;
    auto dst_executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*node_item->node);
    for (const auto &parallel_group : it.second) {
      auto &dependent_nodes = parallel_group_to_nodes_[parallel_group];
      NodeItem *nearest_dep_node = nullptr;
      int max_id = -1;
      for (auto &dep_node : dependent_nodes) {
        if (dep_node->node_id < node_item->node_id && dep_node->node_id > max_id) {
          nearest_dep_node = dep_node;
          max_id = dep_node->node_id;
        }
      }

      if (nearest_dep_node != nullptr) {
        GELOGD("[%s] Nearest node = [%s]", node_item->NodeName().c_str(), nearest_dep_node->NodeName().c_str());
        auto src_engine_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*nearest_dep_node->node);
        if (src_engine_type == dst_executor_type) {
          GELOGD("No need to add dependency for nodes with same executor type");
          continue;
        }
        auto &deps = node_item->dependents_for_execution;
        if (std::find(deps.begin(), deps.end(), nearest_dep_node->node) != deps.end()) {
          GELOGD("%s->%s Already has dependency, skip it",
                 nearest_dep_node->node->GetName().c_str(),
                 node_item->NodeName().c_str());
          continue;
        }
        nearest_dep_node->has_observer = true;
        deps.emplace_back(nearest_dep_node->node);
        GELOGD("Add dependency for nodes with the same parallel group[%s], src = [%s], dst = [%s]",
               parallel_group.c_str(),
               nearest_dep_node->NodeName().c_str(),
               node_item->NodeName().c_str());
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::OptimizeDependenciesForConstantInputs() {
  std::map<NodePtr, std::set<uint32_t>> converted;
  for (auto &it : host_input_value_dependencies_) {
    auto node_item = it.first;
    std::map<NodeItem *, int> ref_counts;
    bool changed = false;
    for (auto output_idx_and_node : it.second) {
      auto output_idx = output_idx_and_node.first;
      auto src_node_item = output_idx_and_node.second;
      ++ref_counts[src_node_item];
      NodePtr constant_node;
      if (src_node_item->node_type == CONSTANT || src_node_item->node_type == CONSTANTOP) {
        constant_node = src_node_item->node;
        GELOGD("src node [%s] is a constant", src_node_item->NodeName().c_str());
      } else {
        auto iter = known_subgraph_constant_output_refs_.find(src_node_item);
        if (iter != known_subgraph_constant_output_refs_.end()) {
          constant_node = iter->second[output_idx];
          if (constant_node != nullptr) {
            GELOGD("Output[%u] of subgraph [%s] is a constant", output_idx, src_node_item->NodeName().c_str());
          }
        }
      }
      if (constant_node == nullptr) {
        GELOGD("Output[%u] of [%s] is not a constant", output_idx, src_node_item->NodeName().c_str());
        continue;
      }
      if (converted[constant_node].count(output_idx) == 0) {
        GE_CHK_STATUS_RET(Convert2HostTensor(constant_node, src_node_item->node_id, output_idx),
                          "[%s] Failed to convert constant to host tensor", constant_node->GetName().c_str());
        converted[constant_node].emplace(output_idx);
      }
      src_node_item->to_const_output_id_list.erase(output_idx);
      --ref_counts[src_node_item];
      changed = true;
    }
    if (changed) {
      std::vector<NodePtr> depends_to_keep;
      for (auto &ref_count_it : ref_counts) {
        if (ref_count_it.second == 0) {
          GELOGD("[%s] no longer depends on [%s] for shape inference",
                 node_item->NodeName().c_str(),
                 ref_count_it.first->NodeName().c_str());
        } else {
          depends_to_keep.emplace_back(ref_count_it.first->node);
        }
      }
      node_item->dependents_for_shape_inference.swap(depends_to_keep);
    }
  }

  return SUCCESS;
}
Status HybridModelBuilder::Convert2HostTensor(const NodePtr &node, int node_id, uint32_t output_idx) {
  auto tensor_value = hybrid_model_.GetTensor(node);
  GE_CHECK_NOTNULL(tensor_value);
  auto tensor_desc = node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_desc);
  Tensor tensor(TensorAdapter::GeTensorDesc2TensorDesc(*tensor_desc));
  int64_t tensor_size = -1;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size),
                          "[%s] Failed to get tensor size", node->GetName().c_str());
  if (tensor_size > 0) {
    auto copy_size = static_cast<size_t>(tensor_size);
    GE_CHECK_GE(tensor_value->GetSize(), copy_size);
    std::vector<uint8_t> buffer(copy_size);
    GE_CHK_RT_RET(rtMemcpy(buffer.data(),
                           copy_size,
                           tensor_value->GetData(),
                           copy_size,
                           RT_MEMCPY_DEVICE_TO_HOST));
    tensor.SetData(std::move(buffer));
    GELOGD("[%s] Copy constant tensor to host successfully, size = %zu", node->GetName().c_str(), copy_size);
  }

  hybrid_model_.host_tensors_[node_id].emplace_back(output_idx, std::move(tensor));
  return SUCCESS;
}

Status HybridModelBuilder::RelinkNextIteration() {
  for (const auto &item : stream_merge_op_nodes_) {
    const auto &merge = item.second;
    std::string node_name;
    if (!AttrUtils::GetStr(merge->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, node_name)) {
      GELOGD("[%s] no attribute[%s], not in while loop", merge->GetName().c_str(), ATTR_NAME_NEXT_ITERATION.c_str());
      continue;
    }

    const auto it = next_iteration_op_nodes_.find(node_name);
    if (it == next_iteration_op_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "[%s] expect NextIteration[%s] not found", merge->GetName().c_str(), node_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &iteration = it->second;
    if (GraphUtils::AddEdge(iteration->GetOutDataAnchor(0), merge->GetInDataAnchor(1)) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[%s] -> [%s] Add edge failed", node_name.c_str(), merge->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::BuildProfilingControl(GraphItem &graph_item,
                                                 const std::map<size_t, std::pair<uint32_t, uint32_t>> &nodes) {
  const auto node_size = graph_item.node_items_.size();
  for (const auto &item : nodes) {
    const auto node_index = item.first;
    GE_CHK_BOOL_RET_STATUS(node_index < node_size, FAILED, "node index invalid");
    const auto &node_item = graph_item.node_items_[node_index];
    if (item.second.first > 0) {
      const auto prev_num = item.second.first;
      if (node_index == prev_num) {
        // Profiling Before root node.
        for (uint32_t i = 1; i <= prev_num; ++i) {
          GE_CHK_BOOL_RET_STATUS(node_index - i < node_size, FAILED, "prev index invalid");
          const auto &curr_item = graph_item.node_items_[node_index - i];
          graph_item.root_items_.emplace(graph_item.root_items_.begin(), curr_item);
        }
      } else {
        GE_CHK_BOOL_RET_STATUS((node_index - prev_num) - 1 < node_size, FAILED, "prev index invalid");
        const auto &prev_item = graph_item.node_items_[(node_index - prev_num) - 1];
        for (uint32_t i = 1; i <= prev_num; ++i) {
          GE_CHK_BOOL_RET_STATUS(node_index - i < node_size, FAILED, "prev index invalid");
          const auto &curr_item = graph_item.node_items_[node_index - i];
          prev_item->SetCtrlSend(curr_item, UINT32_MAX);
          curr_item->SetCtrlSend(node_item, UINT32_MAX);
        }
      }
    }

    if (item.second.second > 0) {
      const auto post_num = item.second.second;
      if (node_size == node_index + post_num + 1) {
        // Profiling After last node.
        for (uint32_t i = 1; i <= post_num; ++i) {
          GE_CHK_BOOL_RET_STATUS(node_index + i < node_size, FAILED, "post index invalid");
          const auto &curr_item = graph_item.node_items_[node_index + i];
          node_item->SetCtrlSend(curr_item, UINT32_MAX);
        }
      } else {
        GE_CHK_BOOL_RET_STATUS((node_index + post_num) + 1 < node_size, FAILED, "post index invalid");
        const auto &post_item = graph_item.node_items_[(node_index + post_num) + 1];
        for (uint32_t i = 1; i <= post_num; ++i) {
          GE_CHK_BOOL_RET_STATUS(node_index + i < node_size, FAILED, "post index invalid");
          const auto &curr_item = graph_item.node_items_[node_index + i];
          node_item->SetCtrlSend(curr_item, UINT32_MAX);
          curr_item->SetCtrlSend(post_item, UINT32_MAX);
        }
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::BuildFrameGroupIndex(NodeItem &node_item) {
  if (node_item.is_root_node_) {
    GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    return SUCCESS;
  }

  int64_t ctrl_flow_group = -1;
  if (node_item.IsEnterOp() && AttrUtils::GetInt(node_item.op_desc, ATTR_NAME_CONTROL_FLOW_GROUP, ctrl_flow_group)) {
    node_item.frame_index_ = ctrl_flow_group;
    for (const auto src_node : node_item.node->GetInAllNodes()) {
      NodeItem *src_node_item = nullptr;
      GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, &src_node_item),
                        "[%s] failed to get or create node item", src_node->GetName().c_str());
      if (!src_node_item->is_root_node_) {
        GELOGD("[%s] frame index: %ld, [%s] parent frame index: %ld", node_item.node_name.c_str(),
               node_item.frame_index_, src_node_item->node_name.c_str(), src_node_item->frame_index_);
        parent_frame_group_[node_item.frame_index_] = src_node_item->frame_index_;
        break;
      }
    }

    const auto it = parent_frame_group_.find(node_item.frame_index_);
    node_item.parent_frame_ = (it != parent_frame_group_.end()) ? it->second : -1;
    GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    return SUCCESS;
  }

  for (const auto src_node : node_item.node->GetInAllNodes()) {
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, &src_node_item),
                      "[%s] failed to get or create node item", src_node->GetName().c_str());
    if (src_node_item->is_root_node_) {
      continue;
    }

    if (src_node_item->IsExitOp()) {
      const auto it = parent_frame_group_.find(src_node_item->frame_index_);
      node_item.frame_index_ = (it != parent_frame_group_.end()) ? it->second : -1;
    } else {
      node_item.frame_index_ = src_node_item->frame_index_;
    }

    const auto it = parent_frame_group_.find(node_item.frame_index_);
    node_item.parent_frame_ = (it != parent_frame_group_.end()) ? it->second : -1;
    GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    return SUCCESS;
  }

  GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
         node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
  return SUCCESS;
}

Status HybridModelBuilder::BuildControlFlowGroup(GraphItem &graph_item, const NodePtr &node, NodeItem *node_item) {
  GELOGD("Build control flow for node %s", node->GetName().c_str());
  using GroupBuilder = std::function<Status(HybridModelBuilder *, const NodePtr &, NodeItem *)>;
  static const std::map<std::string, GroupBuilder> control_flow{
    { STREAMACTIVE, &HybridModelBuilder::CreateStreamActiveGroup },
    { STREAMSWITCH, &HybridModelBuilder::CreateStreamSwitchGroup },
    { STREAMSWITCHN, &HybridModelBuilder::CreateStreamSwitchNGroup },
    { NEXTITERATION, &HybridModelBuilder::CreateNextIterationGroup },
    { REFNEXTITERATION, &HybridModelBuilder::CreateNextIterationGroup },
    { SWITCH, &HybridModelBuilder::CreateSwitchGroup },
    { REFSWITCH, &HybridModelBuilder::CreateSwitchGroup },
    { LABELSET, &HybridModelBuilder::CreateLabelSetGroup },
    { LABELGOTO, &HybridModelBuilder::CreateLabelGotoGroup },
    { LABELGOTOEX, &HybridModelBuilder::CreateLabelGotoGroup },
    { LABELSWITCH, &HybridModelBuilder::CreateLabelSwitchGroup },
    { LABELSWITCHBYINDEX, &HybridModelBuilder::CreateLabelSwitchGroup }
  };

  Status ret = SUCCESS;
  auto it = control_flow.find(node_item->node_type);
  if (it == control_flow.end()) {
    ret = CreateNormalNodeGroup(node, node_item);
  } else {
    graph_item.has_ctrl_flow_op_ = true;
    ret = it->second(this, node, node_item);
  }
  GELOGD("Node: %s, control by: %zu, control for: %zu, switch group: %zu", node->GetName().c_str(),
         node_item->ctrl_recv_.size(), node_item->ctrl_send_.size(), node_item->switch_groups_.size());
  return ret;
}

Status HybridModelBuilder::CreateNormalNodeGroup(const NodePtr &node, NodeItem *node_item) {
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    if ((dst_node->GetType() == STREAMACTIVE) && (kStreamActiveTypes.count(node->GetType()) == 0)) {
      GELOGI("[%s] ignore control to [%s]", node->GetName().c_str(), dst_node->GetName().c_str());
      continue;
    }

    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                      "[%s] failed to get or create node item", dst_node->GetName().c_str());
    node_item->SetCtrlSend(dst_node_item, UINT32_MAX);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateMergeEnterGroup(const NodePtr &node, NodeItem *node_item) {
  // Enter --> StreamActive --> StreamMerge
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    if (dst_node->GetType() != STREAMMERGE) {
      GELOGI("[%s] Skip Not StreamMerge node [%s]", node->GetName().c_str(), dst_node->GetName().c_str());
      continue;
    }
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                      "[%s] failed to get or create node item", dst_node->GetName().c_str());
    // Set Enter Control to StreamMerge as Group 0.
    dst_node_item->switch_groups_.resize(kLoopMergeSize);
    dst_node_item->SetMergeCtrl(node_item, kLoopEnterIdx);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateMergeIterationGroup(const NodePtr &node, NodeItem *node_item) {
  // NextIteration --> StreamActive {-->} StreamMerge
  std::string node_name;
  for (const auto &src_node : node->GetInControlNodes()) {
    GE_CHECK_NOTNULL(src_node);
    if (kNextIterationOpTypes.count(src_node->GetType()) == 0) {
      GELOGI("[%s] Skip Not NextIteration node [%s]", node->GetName().c_str(), src_node->GetName().c_str());
      continue;
    }

    if (!AttrUtils::GetStr(src_node->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, node_name)) {
      GELOGE(INTERNAL_ERROR, "[%s] input node [%s] expect attribute[%s] not found",
             node->GetName().c_str(), src_node->GetName().c_str(), ATTR_NAME_NEXT_ITERATION.c_str());
      return INTERNAL_ERROR;
    }

    const auto it = stream_merge_op_nodes_.find(node_name);
    if (it == stream_merge_op_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "[%s] expect StreamMerge[%s] not found", node->GetName().c_str(), node_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &dst_node = it->second;
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item), "[%s] failed to get or create node item",
                      dst_node->GetName().c_str());
    // Set NextIteration Control to StreamMerge as Group 1.
    dst_node_item->SetMergeCtrl(node_item, kLoopIterationIdx);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateStreamActiveGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != STREAMACTIVE) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node_item->node_type.c_str());
    return INTERNAL_ERROR;
  }

  const auto ctrl_nodes = node->GetInControlNodes();
  if (ctrl_nodes.empty()) {
    GELOGW("Skip no in control node: %s", node->GetName().c_str());
    return SUCCESS;
  }

  const auto IsEnterNode = [](const NodePtr &n) {
    return kEnterOpTypes.count(n->GetType()) > 0;
  };
  const auto IsIterationNode = [](const NodePtr &n) {
    return kNextIterationOpTypes.count(n->GetType()) > 0;
  };

  if (std::any_of(ctrl_nodes.begin(), ctrl_nodes.end(), IsEnterNode)) {
    // Enter --> StreamActive --> StreamMerge
    node_item->is_enter_active_ = true;
    return CreateMergeEnterGroup(node, node_item);
  } else if (std::any_of(ctrl_nodes.begin(), ctrl_nodes.end(), IsIterationNode)) {
    // NextIteration --> StreamActive {-->} StreamMerge
    return CreateMergeIterationGroup(node, node_item);
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateStreamSwitchGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != STREAMSWITCH) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node_item->node_type.c_str());
    return INTERNAL_ERROR;
  }

  // Consider as two groups, group[0] set empty for false, group[1] for true.
  node_item->switch_groups_.resize(kStreamSwitchNum);
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                      "[%s] failed to get or create node item", dst_node->GetName().c_str());
    node_item->SetCtrlSend(dst_node_item, kStreamSwitchIdx);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateStreamSwitchNGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != STREAMSWITCHN) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  uint32_t batch_num = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGE(INTERNAL_ERROR, "[%s] Get ATTR_NAME_BATCH_NUM failed", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (batch_num == 0) {
    GELOGW("[%s] Got empty branch for SwitchN, Please check.", node->GetName().c_str());
    return SUCCESS;
  }

  node_item->switch_groups_.resize(batch_num);
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    std::string batch_label;
    if (!AttrUtils::GetStr(dst_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) {
      GELOGE(INTERNAL_ERROR, "[%s] Get ATTR_NAME_BATCH_LABEL failed", dst_node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    std::string::size_type pos = batch_label.rfind("_");
    if (pos == std::string::npos) {
      GELOGW("[%s] Separator not found in batch label: %s.", dst_node->GetName().c_str(), batch_label.c_str());
      continue;
    }

    ++pos; // Skip Separator
    uint64_t batch_index = std::strtoul(batch_label.data() + pos, nullptr, kDecimal);
    if (batch_index >= batch_num) {
      GELOGW("batch label: %s, batch index: %lu great than batch num: %u", batch_label.c_str(), batch_index, batch_num);
      continue;
    }

    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                      "[%s] failed to get or create node item", dst_node->GetName().c_str());
    node_item->SetCtrlSend(dst_node_item, batch_index);
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateNextIterationGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != NEXTITERATION && node_item->node_type != REFNEXTITERATION) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  return CreateNormalNodeGroup(node, node_item);
}

Status HybridModelBuilder::CreateSwitchGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != SWITCH && node_item->node_type != REFSWITCH) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                      "[%s] failed to get or create node item", dst_node->GetName().c_str());
    node_item->SetCtrlSend(dst_node_item, UINT32_MAX);
  }

  // Group switch flow by out put data.
  node_item->switch_groups_.resize(SWITCH_OUTPUT_NUM);
  for (uint32_t i = 0; i < SWITCH_OUTPUT_NUM; ++i) {
    for (const auto &dst_node : node->GetOutDataNodes()) {
      GE_CHECK_NOTNULL(dst_node);
      NodeItem *dst_node_item = nullptr;
      GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, &dst_node_item),
                        "[%s] failed to get or create node item", dst_node->GetName().c_str());
      node_item->SetCtrlSend(dst_node_item, i); // take switch data as ctrl.
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateLabelSetGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != LABELSET) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGE(UNSUPPORTED, "[%s] Not implemented.", node->GetName().c_str());
  return UNSUPPORTED;
}

Status HybridModelBuilder::CreateLabelGotoGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != LABELGOTO && node_item->node_type != LABELGOTOEX) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGE(UNSUPPORTED, "[%s] Not implemented.", node->GetName().c_str());
  return UNSUPPORTED;
}

Status HybridModelBuilder::CreateLabelSwitchGroup(const NodePtr &node, NodeItem *node_item) {
  if (node_item->node_type != LABELSWITCH && node_item->node_type != LABELSWITCHBYINDEX) {
    GELOGE(INTERNAL_ERROR, "Called by %s is invalid", node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGE(UNSUPPORTED, "[%s] Not implemented.", node->GetName().c_str());
  return UNSUPPORTED;
}
}  // namespace hybrid
}  // namespace ge
