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

#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "cce/aicpu_engine_struct.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "common/dump/dump_manager.h"
#include "common/ge/ge_util.h"
#include "external/graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH, KnownNodeExecutor);

Status KnownNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeTaskExecuteAsync] Start");
  GELOGD("[%s] KnownNodeTask::ExecuteAsync in, model id: %u.", context.GetNodeName(), davinci_model_->Id());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::ExecuteAsync davinci model has no taskinfo.");

    // todo if data is connected to netoutput, forward address ? copy data?
    if (context.NumInputs() == context.NumOutputs()){
      for (int i = 0; i < context.NumInputs(); ++i) {
        auto tensor = context.MutableInput(i);
        GE_CHECK_NOTNULL(tensor);
        GE_CHK_STATUS_RET(context.SetOutput(i, *tensor),
                          "[%s] Failed to set output[%d]",
                          context.GetNodeName(),
                          i);
      }
    }

    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
    return SUCCESS;
  }

  rtError_t rt_ret;
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] Start");
  rt_ret = rtModelExecute(davinci_model_->GetRtModelHandle(), context.GetStream(), 0);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "rtModelExecute error, ret:Ox%X", rt_ret);
                  GELOGE(rt_ret, "[Invoke][rtModelExecute] error, ret:Ox%X", rt_ret);
                  return FAILED;);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodertModelExecute] End");

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(done_callback));
  GELOGD("[%s] KnownNodeTask::ExecuteAsync success, model id: %u.", context.GetNodeName(), davinci_model_->Id());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status KnownNodeTask::UpdateArgs(TaskContext &context) {
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs in.", context.GetNodeName());
  if (davinci_model_->GetTaskList().empty()) {
    GELOGW("KnownNodeExecutor::UpdateArgs davinci model has no taskinfo.");
    return SUCCESS;
  }

  vector<void *> inputs;
  for (int i = 0; i < context.NumInputs(); ++i) {
    TensorValue *tv = context.MutableInput(i);
    GE_CHECK_NOTNULL(tv);
    inputs.emplace_back(tv->MutableData());
  }

  vector<void *> outputs;
  for (int i = 0; i < context.NumOutputs(); ++i) {
    TensorValue *tv = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tv);
    outputs.emplace_back(tv->MutableData());
  }

  GE_CHK_STATUS_RET(davinci_model_->UpdateKnownNodeArgs(inputs, outputs),
                    "[Update][KnownNodeArgs] failed for %s.",  context.GetNodeName());
  GELOGD("[%s] KnownNodeExecutor::UpdateArgs success, task_size = %zu", context.GetNodeName(),
         davinci_model_->GetTaskList().size());
  return SUCCESS;
}

Status KnownNodeTask::Init(TaskContext &context) {
  // allocate output mem
  GE_CHK_STATUS_RET(context.AllocateOutputs(), "[Allocate][Outputs] failed for %s.", context.GetNodeName());
  // allocate mem base
  void *buffer = nullptr;
  int64_t total_useful_size = 0;
  GE_CHK_STATUS_RET(davinci_model_->GetTotalMemSizeExcludeZeroCopy(total_useful_size),
                    "[Get][TotalMemSizeExcludeZeroCopy] failed.");
  if (total_useful_size != 0) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                           "[KnownNodeTask_AllocateWorkspace] Start");
    GE_CHK_STATUS_RET(context.AllocateWorkspace(total_useful_size, &buffer, davinci_model_->GetRuntimeParam().mem_base),
                      "[Allocate][Workspace] failed for %s.", context.GetNodeName());
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                           "[KnownNodeTask_AllocateWorkspace] End, size %ld", total_useful_size);
    // update mem base
    davinci_model_->UpdateMemBase(static_cast<uint8_t *>(buffer));
    GELOGI("KnownNodeTask::Init mem base is %p, size %ld.",
           davinci_model_->GetRuntimeParam().mem_base, total_useful_size);
  }
  GE_CHK_STATUS_RET(ModelManager::GetInstance()->DestroyAicpuKernel(davinci_model_->GetSessionId(),
                                                                    davinci_model_->Id(),
                                                                    davinci_model_->SubModelId()),
                    "[Destroy][AicpuKernel] failed, session_id:%lu, model_id:%u, sub_model_id:%u",
                    davinci_model_->GetSessionId(), davinci_model_->Id(), davinci_model_->SubModelId());
  if (!load_flag_) {
    auto execution_context = const_cast<GraphExecutionContext *>(context.GetExecutionContext());
    GE_CHECK_NOTNULL(execution_context);
    auto &davinci_model = execution_context->davinci_model;
    davinci_model.emplace_back(davinci_model_);
    load_flag_ = true;
  }

  GELOGI("[%s] KnownNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeTask::InitDavinciModel(const HybridModel &model, TensorBuffer *weight_buffer) {
  GELOGD("[Init][DavinciModel] start");
  davinci_model_->InitRuntimeParams();
  GE_CHK_STATUS_RET(davinci_model_->InitVariableMem(), "[Init][VariableMem] failed");
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  davinci_model_->SetDeviceId(static_cast<uint32_t>(device_id));

  auto dump_properties = DumpManager::GetInstance().GetDumpProperties(model.GetSessionId());
  if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
    davinci_model_->SetDumpProperties(dump_properties);
  }

  void *weight = nullptr;
  size_t weight_size = 0;
  if (weight_buffer != nullptr) {
    weight = weight_buffer->GetData();
    weight_size = weight_buffer->GetSize();
  }
  GELOGD("Start to init davinci model, weight size = %zu", weight_size);
  GE_CHK_STATUS_RET(DoInitDavinciModel(weight, weight_size), "[Init][DavinciModel] Failed to init davinci model.");
  GELOGD("[Init][DavinciModel] success");
  return SUCCESS;
}

Status KnownNodeTask::DoInitDavinciModel(void *weight, size_t weight_size) {
  return davinci_model_->Init(nullptr, 0, weight, weight_size);
}

Status KnownNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGD("[%s] KnownNodeExecutor::PrepareTask in.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] Start");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] Start");
  GE_CHK_STATUS_RET(task.Init(context), "[Invoke][Init] %s known node init davinci model failed.",
                    context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorTaskInit] End");

  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] Start");
  GE_CHK_STATUS_RET(task.UpdateArgs(context), "[Invoke][UpdateArgs] %s known node task update args failed.",
                    context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorUpdateArgs] End");
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorPrepareTask] End");
  GELOGD("[%s] KnownNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status KnownNodeExecutor::SetDaviciModel(const HybridModel &model, const NodePtr &node,
                                         std::shared_ptr<DavinciModel> &davinci_model) const {
  // set known node flag as true
  davinci_model->SetKnownNode(true);
  davinci_model->SetId(model.GetModelId());
  davinci_model->SetDumpModelName(model.GetModelName());
  davinci_model->SetOmName(model.GetOmName());
  void *global_step = model.GetGlobalStep();
  GE_CHECK_NOTNULL(global_step);
  davinci_model->SetGlobalStep(global_step, sizeof(int64_t));
  // set model id as root node's node id
  davinci_model->SetSubModelId(node->GetOpDesc()->GetId());
  return SUCCESS;
}

Status KnownNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node,
                                   shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] KnownNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);

  GeModelPtr ge_model;
  ComputeGraphPtr compute_graph;
  GE_CHK_STATUS_RET(GetModelAndGraph(model, node, ge_model, compute_graph),
                    "[%s] Failed to get model and graph",
                    node->GetName().c_str());
  auto node_item = const_cast<NodeItem *>(model.GetNodeItem(node));
  GE_CHECK_NOTNULL(node_item);
  GE_CHK_STATUS_RET_NOLOG(ParseAttrForAllocatingOutputs(*node_item, *compute_graph));
  auto weight_buffer = model.GetModelWeight(compute_graph->GetName());
  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model);

  GE_CHK_STATUS_RET_NOLOG(SetDaviciModel(model, node, davinci_model));
  GELOGD("KnownNodeExecutor::LoadTask node id %ld.", node->GetOpDesc()->GetId());

  GE_CHK_STATUS_RET(davinci_model->Assign(ge_model),
                    "[Invoke][Assign]KnownNodeExecutor::LoadTask davincimodel assign failed for node:%s.",
                    node->GetName().c_str());

  auto known_node_task = MakeShared<KnownNodeTask>(davinci_model);
  GE_CHECK_NOTNULL(known_node_task);
  GE_CHK_STATUS_RET_NOLOG(known_node_task->InitDavinciModel(model, weight_buffer));
  GELOGI("[%s] KnownNodeExecutor::LoadTask success.", node->GetName().c_str());
  task = std::move(known_node_task);
  return SUCCESS;
}

Status KnownNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                      const std::function<void()> &callback) const {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] Start");
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback), "[Invoke][ExecuteAsync]Failed to execute task. node = %s",
                    context.GetNodeItem().NodeName().c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[KnownNodeExecutorExecuteTask] End");
  return SUCCESS;
}

Status KnownNodeExecutor::ParseAttrForAllocatingOutputs(NodeItem &node_item, ComputeGraph &graph) {
  GELOGD("[%s] Start to parse attributes for outputs", node_item.NodeName().c_str());
  auto net_output_node = graph.FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("[%s] Subgraph do not got net output", graph.GetName().c_str());
    return SUCCESS;
  }

  auto net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);
  std::map<std::string, int> connected_inputs;
  std::map<NodePtr, int> data_indices;
  GE_CHK_STATUS_RET(GetDataNodes(graph, data_indices), "[%s] Failed to get data node indices",
                    node_item.NodeName().c_str());
  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    auto src_node = out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto src_op_type = src_node->GetType();
    auto output_index = in_data_anchor->GetIdx();
    GELOGD("Node %s, output %d, src node = %s, src node type = %s",
           node_item.NodeName().c_str(),
           output_index,
           src_node->GetName().c_str(),
           src_op_type.c_str());
    // parse reuse outputs
    std::string input_key = std::to_string(op_desc->GetId()) + "_" + std::to_string(out_data_anchor->GetIdx());
    auto it = connected_inputs.find(input_key);
    if (it == connected_inputs.end()) {
      connected_inputs.emplace(input_key, output_index);
    } else {
      GELOGD("[%s] output [%d] reuse output [%d] input node = %s, idx = %d.", node_item.NodeName().c_str(),
             output_index,
             it->second,
             src_node->GetName().c_str(),
             out_data_anchor->GetIdx());
      node_item.reuse_outputs.emplace(output_index, it->second);
    }

    if (src_op_type == DATA) {
      int data_index = data_indices[src_node];
      node_item.reuse_inputs.emplace(output_index, data_index);
      GELOGD("[%s] output[%u] reuses input[%d]", node_item.NodeName().c_str(), output_index, data_index);
    } else if (src_op_type == CONSTANTOP || src_op_type == CONSTANT || src_op_type == VARIABLE) {
      node_item.ref_outputs.emplace(output_index, src_node);
      GELOGD("[%s] output[%d] ref to node [%s]",
             node_item.NodeName().c_str(),
             output_index,
             src_node->GetName().c_str());
    }
  }

  GELOGD("[%s] Done parsing attributes for outputs successfully", node_item.NodeName().c_str());
  return SUCCESS;
}

Status KnownNodeExecutor::GetDataNodes(ComputeGraph &graph, std::map<NodePtr, int> &data_indices) {
  std::map<int, NodePtr> ordered_data_nodes;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == DATA) {
      int index = -1;
      (void) AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      ordered_data_nodes.emplace(index, node);
    }
  }

  // reindex
  int data_index = 0;
  for (const auto &it : ordered_data_nodes) {
    data_indices.emplace(it.second, data_index++);
  }

  return SUCCESS;
}

Status KnownNodeExecutor::GetModelAndGraph(const HybridModel &model,
                                           const NodePtr &node,
                                           GeModelPtr &ge_model,
                                           ComputeGraphPtr &graph) {
  ge_model = model.GetGeModel(node);
  GE_CHECK_NOTNULL(ge_model);
  const auto &root_graph = model.GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  AscendString graph_name;
  GE_CHK_GRAPH_STATUS_RET(ge_model->GetGraph().GetName(graph_name), "Failed to get subgraph name");
  graph = root_graph->GetSubgraph(graph_name.GetString());
  GE_CHECK_NOTNULL(graph);
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
