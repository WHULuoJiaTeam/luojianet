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

#include "hybrid/executor/subgraph_executor.h"
#include "graph/ge_context.h"
#include "hybrid/executor/worker/task_compile_engine.h"
#include "hybrid/executor/worker/execution_engine.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int kDefaultThreadNum = 4;
constexpr int kDefaultQueueSize = 16;
constexpr int kDataInputIndex = 0;
}

SubgraphExecutor::SubgraphExecutor(const GraphItem *graph_item, GraphExecutionContext *context, bool force_infer_shape,
                                   ThreadPool *pre_run_pool)
    : graph_item_(graph_item),
      context_(context),
      force_infer_shape_(force_infer_shape),
      pre_run_pool_(pre_run_pool),
      own_thread_pool_(false),
      ready_queue_(kDefaultQueueSize) {
}

SubgraphExecutor::~SubgraphExecutor() {
  if (own_thread_pool_ && pre_run_pool_ != nullptr) {
    delete pre_run_pool_;
  }
  GELOGD("[%s] SubgraphExecutor destroyed.", graph_item_->GetName().c_str());
}

Status SubgraphExecutor::Init(const std::vector<TensorValue> &inputs,
                              const std::vector<ConstGeTensorDescPtr> &input_desc) {
  if (pre_run_pool_ == nullptr) {
    pre_run_pool_ = new (std::nothrow) ThreadPool(kDefaultThreadNum);
    GE_CHECK_NOTNULL(pre_run_pool_);
    own_thread_pool_ = true;
  }
  subgraph_context_.reset(new(std::nothrow)SubgraphContext(graph_item_, context_));
  GE_CHECK_NOTNULL(subgraph_context_);
  GE_CHK_STATUS_RET(subgraph_context_->Init(),
      "[Init][SubgraphContext][%s] Failed to init subgraph context.", graph_item_->GetName().c_str());

  shape_inference_engine_.reset(new(std::nothrow) ShapeInferenceEngine(context_, subgraph_context_.get()));
  GE_CHECK_NOTNULL(shape_inference_engine_);

  if (graph_item_->IsDynamic()) {
    GE_CHK_STATUS_RET(InitInputsForUnknownShape(inputs, input_desc),
                      "[%s] Failed to set inputs.",
                      graph_item_->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitInputsForKnownShape(inputs),
        "[Invoke][InitInputsForKnownShape][%s] Failed to init subgraph executor for known shape subgraph.",
        graph_item_->GetName().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::InitInputsForUnknownShape(const std::vector<TensorValue> &inputs,
                                                   const std::vector<ConstGeTensorDescPtr> &input_desc) {
  // Number of inputs of parent node should be greater or equal than that of subgraph
  auto input_nodes = graph_item_->GetInputNodes();
  if (inputs.size() < input_nodes.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size][%s] Number of inputs [%zu] is not sufficient for subgraph which needs [%zu] inputs.",
           graph_item_->GetName().c_str(), inputs.size(), input_nodes.size());
    REPORT_INNER_ERROR("E19999",
                       "[%s] Number of inputs [%zu] is not sufficient for subgraph which needs [%zu] inputs.",
                       graph_item_->GetName().c_str(), inputs.size(), input_nodes.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0; i < input_nodes.size(); ++i) {
    auto &input_node = input_nodes[i];
    if (input_node == nullptr) {
      GELOGD("[%s] Input[%zu] is not needed by subgraph, skip it.", graph_item_->GetName().c_str(), i);
      continue;
    }

    auto &input_tensor = inputs[i];
    GELOGD("[%s] Set input tensor[%zu] to inputs with index = %d, tensor = %s",
           graph_item_->GetName().c_str(),
           i,
           input_node->input_start,
           input_tensor.DebugString().c_str());

    GE_CHK_STATUS_RET(subgraph_context_->SetInput(*input_node, kDataInputIndex, input_tensor),
                      "[Invoke][SetInput] failed for grap_item[%s] input tensor[%zu]",
                      graph_item_->GetName().c_str(), i);

    if (force_infer_shape_ || input_node->is_dynamic) {
      GELOGD("[%s] Start to update input[%zu] for subgraph data node.", graph_item_->GetName().c_str(), i);
      GE_CHECK_LE(i + 1, input_desc.size());
      const auto &tensor_desc = input_desc[i];
      GE_CHECK_NOTNULL(tensor_desc);
      auto node_state = subgraph_context_->GetOrCreateNodeState(input_node);
      GE_CHECK_NOTNULL(node_state);
      node_state->GetShapeInferenceState().UpdateInputShape(0, *tensor_desc);
      auto op_desc = input_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      auto output_desc = op_desc->MutableOutputDesc(kDataInputIndex);
      GE_CHECK_NOTNULL(output_desc);
      output_desc->SetShape(tensor_desc->GetShape());
      output_desc->SetOriginShape(tensor_desc->GetOriginShape());
      node_state->SetSkipInferShape(true);
    }
  }

  GELOGD("[%s] Done setting inputs.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::InitInputsForKnownShape(const std::vector<TensorValue> &inputs) {
  auto &input_index_mapping = graph_item_->GetInputIndexMapping();
  for (size_t i = 0; i < input_index_mapping.size(); ++i) {
    auto &parent_input_index = input_index_mapping[i];
    if (static_cast<size_t>(parent_input_index) >= inputs.size()) {
      GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Number of inputs [%zu] is not sufficient for subgraph"
             "which needs at lease [%d] inputs", graph_item_->GetName().c_str(), inputs.size(),
             parent_input_index + 1);
      REPORT_INNER_ERROR("E19999", "[%s] Number of inputs [%zu] is not sufficient for subgraph"
                         "which needs at lease [%d] inputs",
                         graph_item_->GetName().c_str(), inputs.size(), parent_input_index + 1);
      return INTERNAL_ERROR;
    }

    auto &input_tensor = inputs[parent_input_index];
    subgraph_context_->SetInput(static_cast<int>(i), input_tensor);
    GELOGD("[%s] Set input tensor[%zu] with inputs with index = %d, tensor = %s",
           graph_item_->GetName().c_str(),
           i,
           parent_input_index,
           input_tensor.DebugString().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const std::vector<TensorValue> &inputs,
                                      const std::vector<ConstGeTensorDescPtr> &input_desc,
                                      const std::vector<TensorValue> &outputs) {
  GELOGD("[%s] is dynamic = %s", graph_item_->GetName().c_str(), graph_item_->IsDynamic() ? "true" : "false");
  GE_CHK_STATUS_RET(Init(inputs, input_desc), "[Invoke][Init]failed for [%s].", graph_item_->GetName().c_str());
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET(EnableOutputZeroCopy(outputs),
                      "[Invoke][EnableOutputZeroCopy] Failed by user provided outputs.");
  }
  if (!graph_item_->IsDynamic()) {
    return ExecuteAsyncForKnownShape(inputs);
  }

  HYBRID_CHK_STATUS_RET(ScheduleTasks(), "[%s] Failed to execute tasks.", graph_item_->GetName().c_str());
  GELOGD("[%s] Done executing subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const std::vector<TensorValue> &inputs,
                                      const std::vector<ConstGeTensorDescPtr> &input_desc) {
  return ExecuteAsync(inputs, input_desc, {});
}

Status SubgraphExecutor::ExecuteAsyncForKnownShape(const std::vector<TensorValue> &inputs) {
  GELOGD("[%s] subgraph is not dynamic.", graph_item_->GetName().c_str());
  if (graph_item_->GetAllNodes().size() != 1) {
    REPORT_INNER_ERROR("E19999", "[%s] Invalid known shape subgraph. node size = %zu",
                       graph_item_->GetName().c_str(), graph_item_->GetAllNodes().size());
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Invalid known shape subgraph. node size = %zu",
           graph_item_->GetName().c_str(), graph_item_->GetAllNodes().size());
    return INTERNAL_ERROR;
  }

  auto node_item = graph_item_->GetAllNodes()[0];
  GE_CHECK_NOTNULL(node_item);
  auto node_state = subgraph_context_->GetOrCreateNodeState(node_item);
  GE_CHECK_NOTNULL(node_state);
  node_state->SetKernelTask(node_item->kernel_task);

  std::function<void()> callback;
  GE_CHK_STATUS_RET_NOLOG(InitCallback(node_state.get(), callback));
  HYBRID_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, node_state->GetTaskContext(), *context_, callback),
                        "[%s] Failed to execute node [%s] for known subgraph.",
                        graph_item_->GetName().c_str(),
                        node_state->GetName().c_str());

  GELOGD("[%s] Done execute non-dynamic subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(TaskContext &task_context) {
  std::vector<TensorValue> inputs;
  std::vector<ConstGeTensorDescPtr> input_desc;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    auto tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    inputs.emplace_back(*tensor);
    input_desc.emplace_back(task_context.GetInputDesc(i));
  }

  GE_CHK_STATUS_RET(ExecuteAsync(inputs, input_desc), "[Invoke][ExecuteAsync] failed for [%s].",
                    graph_item_->GetName().c_str());

  GE_CHK_STATUS_RET(SetOutputsToParentNode(task_context),
                    "[Invoke][SetOutputsToParentNode][%s] Failed to set output shapes to parent node.",
                    graph_item_->GetName().c_str());
  return SUCCESS;
}

BlockingQueue<const NodeItem *> &SubgraphExecutor::GetPrepareQueue(int group) {
  std::lock_guard<std::mutex> lk(mu_);
  return prepare_queues_[group];
}

Status SubgraphExecutor::NodeEnqueue(NodeState *node_state) {
  if (!ready_queue_.Push(node_state)) {
    if (context_->is_eos_) {
      GELOGD("Got end of sequence");
      return SUCCESS;
    }
    GELOGE(INTERNAL_ERROR, "[Check][State][%s] Error occurs while launching tasks. quit from preparing nodes.",
           graph_item_->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Error occurs while launching tasks. quit from preparing nodes.",
                       graph_item_->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("[%s] Push node [%s] to queue.", graph_item_->GetName().c_str(), node_state->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::PrepareNode(const NodeItem &node_item, int group) {
  GELOGD("[%s] Start to prepare node [%s].", graph_item_->GetName().c_str(), node_item.NodeName().c_str());
  // for while op
  if (force_infer_shape_ && !node_item.is_dynamic) {
    GELOGD("[%s] Force infer shape is set, updating node to dynamic.", node_item.NodeName().c_str());
    auto &mutable_node_item = const_cast<NodeItem &>(node_item);
    mutable_node_item.SetToDynamic();
  }

  auto node_state = subgraph_context_->GetOrCreateNodeState(&node_item);
  GE_CHECK_NOTNULL(node_state);
  auto p_node_state = node_state.get();

  if (node_item.node_type == NETOUTPUT) {
    GE_CHK_STATUS_RET_NOLOG(NodeEnqueue(p_node_state));
    return AfterPrepared(p_node_state);
  }

  // only do shape inference and compilation for nodes with dynamic shapes.
  if (node_item.is_dynamic) {
    GE_CHECK_NOTNULL(pre_run_pool_);
    auto prepare_future = pre_run_pool_->commit([this, p_node_state]() -> Status {
      GetContext().SetSessionId(context_->session_id);
      GetContext().SetContextId(context_->context_id);
      GE_CHK_STATUS_RET_NOLOG(InferShape(shape_inference_engine_.get(), *p_node_state));
      GE_CHK_STATUS_RET_NOLOG(PrepareForExecution(context_, *p_node_state));
      return AfterPrepared(p_node_state);
    });

    p_node_state->SetPrepareFuture(std::move(prepare_future));
    return NodeEnqueue(p_node_state);
  } else {
    GELOGD("[%s] Skipping shape inference and compilation for node with static shape.",
           node_item.NodeName().c_str());
    if (node_item.kernel_task == nullptr) {
      GELOGW("[%s] Node of static shape got no task.", node_item.NodeName().c_str());
      GE_CHK_STATUS_RET(TaskCompileEngine::Compile(*p_node_state, context_),
                        "[Invoke][Compile] failed for [%s].", p_node_state->GetName().c_str());
    } else {
      node_state->SetKernelTask(node_item.kernel_task);
    }
    const auto &task = node_state->GetKernelTask();
    if (task == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Get][KernelTask] failed for[%s], NodeTask is null.", node_state->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "GetKernelTask failed for %s, nodetask is null.", node_state->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHK_STATUS_RET_NOLOG(NodeEnqueue(p_node_state));
    return AfterPrepared(p_node_state);
  }
}

Status SubgraphExecutor::PrepareNodes(int group) {
  const size_t node_size = graph_item_->GetNodeSize(group);
  GELOGD("[%s] Start to prepare nodes. group = %d, size = %zu", graph_item_->GetName().c_str(), group, node_size);
  if (!graph_item_->HasCtrlFlowOp()) {
    for (const auto &node_item : graph_item_->GetAllNodes(group)) {
      RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
      GE_CHK_STATUS_RET(PrepareNode(*node_item, group), "[%s] failed to prepare task.", node_item->NodeName().c_str());
      RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    }

    GELOGD("[%s] Done preparing nodes successfully.", graph_item_->GetName().c_str());
    return SUCCESS;
  }

  // Initialize the ready queue
  size_t node_count = 0;
  bool node_complete = false;
  for (const auto &node_item : graph_item_->GetRootNodes(group)) {
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
    GE_CHK_STATUS_RET(PrepareNode(*node_item, group), "[%s] failed to prepare task.", node_item->NodeName().c_str());
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    node_complete = node_item->NodeType() == NETOUTPUT;
    node_count++;
  }

  GELOGD("[%s] Done preparing root nodes.", graph_item_->GetName().c_str());
  BlockingQueue<const NodeItem *> &prepare_queue = GetPrepareQueue(group);
  while (((group != -1) && (node_count < node_size)) || ((group == -1) && !node_complete)) {
    const NodeItem *node_item = nullptr;
    if (!prepare_queue.Pop(node_item)) {
      if (context_->is_eos_) {
        GELOGD("[%s] Got end of sequence.", graph_item_->GetName().c_str());
        break;
      }
      if (context_->GetStatus() != SUCCESS) {
        GELOGD("[%s] Graph execution Got failed.", graph_item_->GetName().c_str());
        return SUCCESS;
      }
      GELOGE(INTERNAL_ERROR, "[%s] failed to pop node.", graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (node_item == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      break;
    }

    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
    GE_CHK_STATUS_RET(PrepareNode(*node_item, group), "[%s] failed to prepare task.", node_item->NodeName().c_str());
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    node_complete = node_item->NodeType() == NETOUTPUT;
    node_count++;
  }

  GELOGD("[%s] Done preparing nodes successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::NodeScheduled(NodeState *node_state) {
  GELOGD("Graph[%s] After [%s] scheduled, data size: %zu, ctrl size: %zu, switch index: %d, merge index: %d",
         graph_item_->GetName().c_str(), node_state->GetName().c_str(),
         node_state->GetNodeItem()->data_send_.size(), node_state->GetNodeItem()->ctrl_send_.size(),
         node_state->GetSwitchIndex(), node_state->GetMergeIndex());

  GE_CHECK_NOTNULL(pre_run_pool_);
  auto future = pre_run_pool_->commit([this, node_state]() -> Status {
    RECORD_CALLBACK_EVENT(context_, node_state->GetName().c_str(), "[NodeScheduled] Start");
    std::function<void(const NodeItem *)> callback = [&](const NodeItem *node_item) {
      const auto &node_name = node_item->node_name;
      int group = (node_state->GetGroup() != -1) ? node_item->group : -1;
      GELOGI("After [%s] scheduled, [%s] is ready for prepare.", node_state->GetName().c_str(), node_name.c_str());
      BlockingQueue<const NodeItem *> &prepare_queue = GetPrepareQueue(group);
      if (!prepare_queue.Push(node_item)) {
        if (!context_->is_eos_) {
          GELOGE(INTERNAL_ERROR, "[Check][State][%s] error occurs when push to queue.", graph_item_->GetName().c_str());
          REPORT_INNER_ERROR("E19999", "[%s] error occurs when push to queue.", graph_item_->GetName().c_str());
        }
      }
    };

    GE_CHK_STATUS_RET_NOLOG(node_state->NodeScheduled(callback));
    RECORD_CALLBACK_EVENT(context_, node_state->GetName().c_str(), "[NodeScheduled] End");
    return SUCCESS;
  });

  node_state->SetScheduleFuture(std::move(future));
  if (schedule_queue_.Push(node_state)) {
    return SUCCESS;
  }

  if (context_->is_eos_) {
    GELOGD("[%s] Got end of sequence", graph_item_->GetName().c_str());
    return SUCCESS;
  }

  GELOGE(INTERNAL_ERROR, "[Check][State][%s] error occurs when push to queue.", graph_item_->GetName().c_str());
  REPORT_INNER_ERROR("E19999", "[%s] error occurs when push to queue.", graph_item_->GetName().c_str());
  return INTERNAL_ERROR;
}

Status SubgraphExecutor::AfterPrepared(NodeState *node_state) {
  if (!graph_item_->HasCtrlFlowOp()) {
    return SUCCESS;
  }
  if (node_state->IsShapeDependence()) {
    return SUCCESS;
  }

  // Not control flow node, propagate state.
  return NodeScheduled(node_state);
}

void SubgraphExecutor::AfterExecuted(NodeState *node_state) {
  if (!node_state->IsShapeDependence()) {
    return;
  }

  // For control flow node, propagate state.
  auto error = NodeScheduled(node_state);
  if (error != SUCCESS) {
    auto task_context = node_state->GetTaskContext();
    task_context->OnError(error);
  }
}

void SubgraphExecutor::OnNodeDone(NodeState *node_state) {
  auto task_context = node_state->GetTaskContext();
  NodeDoneCallback cb(context_, task_context);
  auto error = cb.OnNodeDone();
  if (error != SUCCESS) {
    task_context->OnError(error);
  }

  if (node_state->IsShapeDependence() && graph_item_->HasCtrlFlowOp()) {
    AfterExecuted(node_state);
  }
}

Status SubgraphExecutor::InitCallback(NodeState *node_state, std::function<void()> &callback) {
  auto task_context = node_state->GetTaskContext();
  GE_CHECK_NOTNULL(task_context);
  if (task_context->NeedCallback()) {
    callback = std::bind(&SubgraphExecutor::OnNodeDone, this, node_state);
  } else if (node_state->IsShapeDependence() && graph_item_->HasCtrlFlowOp()) {
    callback = std::bind(&SubgraphExecutor::AfterExecuted, this, node_state);
  }

  return SUCCESS;
}

Status SubgraphExecutor::ScheduleNodes() {
  GELOGD("[%s] Start to schedule nodes.", graph_item_->GetName().c_str());
  while (true) {
    NodeState *node_state = nullptr;
    if (!schedule_queue_.Pop(node_state)) {
      if (context_->is_eos_) {
        GELOGD("[%s] Got end of sequence.", graph_item_->GetName().c_str());
        break;
      }
      if (context_->GetStatus() != SUCCESS) {
        GELOGD("[%s] Graph execution Got failed.", graph_item_->GetName().c_str());
        return SUCCESS;
      }
      GELOGE(INTERNAL_ERROR, "[%s] failed to pop node.", graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (node_state == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      break;
    }

    GE_CHK_STATUS_RET_NOLOG(node_state->WaitForScheduleDone());
  }

  GELOGD("[%s] Done schedule nodes successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::InferShape(ShapeInferenceEngine *shape_inference_engine, NodeState &node_state) const {
  HYBRID_CHK_STATUS_RET(shape_inference_engine->InferShape(node_state),
                        "[Invoke][InferShape] failed for [%s].", node_state.GetName().c_str());
  HYBRID_CHK_STATUS_RET(shape_inference_engine->PropagateOutputShapes(node_state),
                        "[Invoke][PropagateOutputShapes] failed for [%s].", node_state.GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::PrepareForExecution(GraphExecutionContext *ctx, NodeState &node_state) {
  auto &node_item = *node_state.GetNodeItem();
  if (node_item.kernel_task == nullptr) {
    GE_CHK_STATUS_RET(TaskCompileEngine::Compile(node_state, ctx),
                      "[Invoke][Compile] Failed for node[%s]", node_state.GetName().c_str());
  } else {
    node_state.SetKernelTask(node_item.kernel_task);
  }
  const auto &task = node_state.GetKernelTask();
  if (task == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][GetKernelTask] failed for[%s], NodeTask is null.", node_state.GetName().c_str());
    REPORT_CALL_ERROR("E19999", "invoke GetKernelTask failed for %s, NodeTask is null.", node_state.GetName().c_str());
    return INTERNAL_ERROR;
  }
  GE_CHK_RT_RET(rtCtxSetCurrent(ctx->rt_context));
  RECORD_COMPILE_EVENT(ctx, node_item.NodeName().c_str(), "[UpdateTilingData] start");
  GE_CHK_STATUS_RET_NOLOG(task->UpdateTilingData(*node_state.GetTaskContext())); // update op_desc before alloc ws
  RECORD_COMPILE_EVENT(ctx, node_item.NodeName().c_str(), "[UpdateTilingData] end");
  return SUCCESS;
}

Status SubgraphExecutor::LaunchTasks() {
  while (true) {
    NodeState *node_state = nullptr;
    if (!ready_queue_.Pop(node_state)) {
      GELOGE(INTERNAL_ERROR, "[Invoke][Pop] failed for [%s].", graph_item_->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "invoke pop failed for %s.", graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (node_state == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      return SUCCESS;
    }

    if (node_state->GetType() == NETOUTPUT) {
      // Wait for all inputs become valid
      // after PrepareNodes returned. all output tensors and shapes are valid
      GE_CHK_STATUS_RET_NOLOG(node_state->GetShapeInferenceState().AwaitShapesReady(*context_));
      GE_CHK_STATUS_RET_NOLOG(node_state->AwaitInputTensors(*context_));
      GELOGD("[%s] Done executing node successfully.", node_state->GetName().c_str());
      continue;
    }

    GE_CHK_STATUS_RET_NOLOG(node_state->WaitForPrepareDone());

    GELOGD("[%s] Start to execute.", node_state->GetName().c_str());
    auto shared_task_context = node_state->GetTaskContext();
    GE_CHECK_NOTNULL(shared_task_context);
    shared_task_context->SetForceInferShape(force_infer_shape_);

    std::function<void()> callback;
    GE_CHK_STATUS_RET_NOLOG(InitCallback(node_state, callback));
    HYBRID_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, shared_task_context, *context_, callback),
                          "[Invoke][ExecuteAsync] failed for [%s].", node_state->GetName().c_str());
    GELOGD("[%s] Done executing node successfully.", node_state->GetName().c_str());
  }
}

Status SubgraphExecutor::ScheduleTasks(int group) {
  GELOGD("[%s] Start to schedule prepare workers.", graph_item_->GetName().c_str());
  subgraph_context_->SetGroup(group);
  auto prepare_future = std::async(std::launch::async, [&]() -> Status {
    GetContext().SetSessionId(context_->session_id);
    GetContext().SetContextId(context_->context_id);
    auto ret = PrepareNodes(group);
    ready_queue_.Push(nullptr);
    schedule_queue_.Push(nullptr);
    for (auto &item : prepare_queues_) {
      item.second.Push(nullptr);
    }
    return ret;
  });

  auto schedule_future = std::async(std::launch::async, [&]() -> Status {
    return ScheduleNodes();
  });

  GELOGD("[%s] Start to execute subgraph.", graph_item_->GetName().c_str());
  auto ret = LaunchTasks();
  if (ret != SUCCESS) {
    subgraph_context_->OnError(ret);
    context_->SetErrorCode(ret);
    ready_queue_.Stop();
    schedule_queue_.Stop();
    for (auto &item : prepare_queues_) {
      item.second.Stop();
    }
    prepare_future.wait();
    schedule_future.wait();
    return ret;
  }

  GE_CHK_STATUS_RET(prepare_future.get(), "[Invoke][get] [%s] Error occurred in task preparation.",
                    graph_item_->GetName().c_str());

  GE_CHK_STATUS_RET(schedule_future.get(), "[Invoke][get] [%s] Error occurred in task preparation.",
                    graph_item_->GetName().c_str());

  GELOGD("[%s] Done launching all tasks successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::GetOutputs(vector<TensorValue> &outputs) {
  return subgraph_context_->GetOutputs(outputs);
}

Status SubgraphExecutor::GetOutputs(vector<TensorValue> &outputs, std::vector<ConstGeTensorDescPtr> &output_desc) {
  GE_CHK_STATUS_RET(GetOutputs(outputs), "[Invoke][GetOutputs] failed for [%s].", graph_item_->GetName().c_str());

  // copy output data from op to designated position
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc),
                    "[Invoke][GetOutputDescList][%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());
  if (outputs.size() != output_desc.size()) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Number of outputs(%zu) mismatch number of output_desc(%zu).",
           outputs.size(), output_desc.size());
    REPORT_INNER_ERROR("E19999", "Number of outputs(%zu) mismatch number of output_desc(%zu).",
                       outputs.size(), output_desc.size());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status SubgraphExecutor::Synchronize() {
  GELOGD("[%s] Synchronize start.", graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(context_->Synchronize(context_->stream));
  GELOGD("[%s] Done synchronizing successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::SetOutputsToParentNode(TaskContext &task_context) {
  // get output tensors and tensor desc list
  std::vector<TensorValue> outputs;
  std::vector<ConstGeTensorDescPtr> output_desc_list;
  GE_CHK_STATUS_RET(subgraph_context_->GetOutputs(outputs), "[Invoke][GetOutputs][%s] Failed to get output tensors.",
                    graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc_list),
                    "[Invoke][GetOutputDescList][%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());

  if (outputs.size() != output_desc_list.size()) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] num of output tensors = %zu, num of output tensor desc = %zu not equal",
           graph_item_->GetName().c_str(), outputs.size(), output_desc_list.size());
    REPORT_INNER_ERROR("E19999", "%s num of output tensors = %zu, num of output tensor desc = %zu not equal",
                       graph_item_->GetName().c_str(), outputs.size(), output_desc_list.size());
    return INTERNAL_ERROR;
  }

  // mapping to parent task context
  for (size_t i = 0; i < outputs.size(); ++i) {
    int parent_output_index = graph_item_->GetParentOutputIndex(i);
    GE_CHECK_GE(parent_output_index, 0);
    // update tensor
    GELOGD("[%s] Updating output[%zu] to parent output[%d]",
           graph_item_->GetName().c_str(),
           i,
           parent_output_index);

    GELOGD("[%s] Updating output tensor, index = %d, tensor = %s",
           graph_item_->GetName().c_str(),
           parent_output_index,
           outputs[i].DebugString().c_str());
    GE_CHK_STATUS_RET(task_context.SetOutput(parent_output_index, outputs[i]));

    // updating shapes. dynamic format/dtype is not supported.
    // It should be noted that even the subgraph is of known shape, it is also necessary to update parent output desc,
    // for instance, IfOp may have two known-shaped subgraphs of different output shapes
    const auto &output_desc = output_desc_list[i];
    auto parent_output_desc = task_context.MutableOutputDesc(parent_output_index);
    GE_CHECK_NOTNULL(parent_output_desc);
    GELOGD("[%s] Updating output shape[%d] from [%s] to [%s]",
           graph_item_->GetName().c_str(),
           parent_output_index,
           parent_output_desc->MutableShape().ToString().c_str(),
           output_desc->GetShape().ToString().c_str());
    parent_output_desc->SetShape(output_desc->GetShape());

    GELOGD("[%s] Updating output original shape[%d] from [%s] to [%s]",
           graph_item_->GetName().c_str(),
           parent_output_index,
           parent_output_desc->GetOriginShape().ToString().c_str(),
           output_desc->GetOriginShape().ToString().c_str());
    parent_output_desc->SetOriginShape(output_desc->GetOriginShape());
  }

  return SUCCESS;
}

Status SubgraphExecutor::EnableOutputZeroCopy(const vector<TensorValue> &outputs) {
  GELOGD("To enable zero copy, output number = %zu", outputs.size());
  const auto &output_edges = graph_item_->GetOutputEdges();
  // Op -> MetOutput, set the output tensor of Op that output to the NetOutput node
  if (outputs.size() != output_edges.size()) {
    GELOGE(PARAM_INVALID, "[Check][Size]Output number mismatches, expect = %zu, but given = %zu",
           output_edges.size(), outputs.size());
    REPORT_INNER_ERROR("E19999", "Output number mismatches, expect = %zu, but given = %zu",
                       output_edges.size(), outputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &output_tensor = outputs[i];
    auto &output_node = output_edges[i].first;
    int output_idx = output_edges[i].second;
    GELOGD("[%s] Set output tensor[%zu] to [%s]'s output[%d], tensor = %s",
           graph_item_->GetName().c_str(),
           i,
           output_node->NodeName().c_str(),
           output_idx,
           output_tensor.DebugString().c_str());

    GE_CHK_STATUS_RET(subgraph_context_->SetOutput(*output_node, output_idx, output_tensor),
                      "[Invoke][SetOutput][%s] Failed to set input tensor[%zu]",
                      graph_item_->GetName().c_str(), i);
  }

  GELOGD("Done enabling zero copy for outputs successfully.");
  return SUCCESS;
}

Status SubgraphExecutor::PartialExecuteAsync(int task_group) {
  return ScheduleTasks(task_group);
}

Status SubgraphExecutor::InitForPartialExecution(const vector<TensorValue> &inputs,
                                                 const vector<ConstGeTensorDescPtr> &input_desc) {
  if (subgraph_context_ == nullptr) {
    return Init(inputs, input_desc);
  }
  subgraph_context_->Reset();
  if (graph_item_->IsDynamic()) {
    GE_CHK_STATUS_RET(InitInputsForUnknownShape(inputs, input_desc),
                      "[%s] Failed to set inputs.",
                      graph_item_->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitInputsForKnownShape(inputs),
                      "[Invoke][InitInputsForKnownShape][%s] Failed to init subgraph executor for known shape subgraph",
                      graph_item_->GetName().c_str());
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
