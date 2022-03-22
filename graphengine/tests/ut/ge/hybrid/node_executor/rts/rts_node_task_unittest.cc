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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/rts/rts_node_executor.h"
#include "common/model/ge_root_model.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestRtsNodeTask : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

TEST_F(UtestRtsNodeTask, test_stream_switch_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "switch", STREAMSWITCH, 2, 0);
  ASSERT_TRUE(AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND, 0));

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 110;
  uint64_t value_1 = 120;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  TensorValue in_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);
  subgraph_context.SetInput(*node_item, 1, in_tensor1);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(node_state->GetSwitchIndex(), -1);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_state->GetSwitchIndex(), 0); // not equal, active 0

  uint64_t value_2 = 110;
  TensorValue in_tensor2(&value_2, sizeof(value_2));
  subgraph_context.SetInput(*node_item, 1, in_tensor2);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_state->GetSwitchIndex(), 1); // equal, active 1
}

TEST_F(UtestRtsNodeTask, test_stream_active_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "active", STREAMACTIVE, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(node_state->GetSwitchIndex(), -1);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_state->GetSwitchIndex(), -1);

  node_item->ctrl_send_.emplace(nullptr);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_state->GetSwitchIndex(), 0);
}

TEST_F(UtestRtsNodeTask, test_stream_merge_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 110;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);
  uint64_t value_1 = 220;
  TensorValue in_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetInput(*node_item, 1, in_tensor1);

  uint64_t value_2 = 123;
  TensorValue out_tensor0(&value_2, sizeof(value_2));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);
  uint64_t value_3 = 223;
  TensorValue out_tensor1(&value_3, sizeof(value_3));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  node_state->SetMergeIndex(1);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_state->GetSwitchIndex(), -1);

  uint64_t value_4 = 323;
  ASSERT_EQ(node_state->GetTaskContext()->GetOutput(0)->CopyScalarValueToHost(value_4), SUCCESS);
  ASSERT_EQ(value_4, value_1);

  uint64_t value_5 = 423;
  ASSERT_EQ(node_state->GetTaskContext()->GetOutput(1)->CopyScalarValueToHost(value_5), SUCCESS);
  ASSERT_EQ(value_5, 1);
}

TEST_F(UtestRtsNodeTask, test_memcpy_async_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "memcpy", MEMCPYASYNC, 1, 1);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 110;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);

  uint64_t value_1 = 123;
  TensorValue out_tensor0(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  uint64_t value_4 = 323;
  ASSERT_EQ(node_state->GetTaskContext()->GetOutput(0)->CopyScalarValueToHost(value_4), SUCCESS);
  ASSERT_EQ(value_4, value_0);
  ASSERT_EQ(value_1, value_0);
}

TEST_F(UtestRtsNodeTask, test_pass_through_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "enter", ENTER, 1, 1);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 110;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);

  uint64_t value_1 = 123;
  TensorValue out_tensor0(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  uint64_t value_4 = 323;
  ASSERT_EQ(node_state->GetTaskContext()->GetOutput(0)->CopyScalarValueToHost(value_4), SUCCESS);
  ASSERT_EQ(value_4, value_0);
}

TEST_F(UtestRtsNodeTask, test_unsupport_label_set) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "labelset", LABELSET, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 2;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), UNSUPPORTED);
}

TEST_F(UtestRtsNodeTask, test_unsupport_label_goto) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "labelgoto", LABELGOTO, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 2;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), UNSUPPORTED);
}

TEST_F(UtestRtsNodeTask, test_unsupport_label_switch) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "labelswitch", LABELSWITCH, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 2;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  RtsNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), UNSUPPORTED);
}
} // namespace ge