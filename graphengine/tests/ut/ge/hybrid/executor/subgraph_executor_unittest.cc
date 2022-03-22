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
#include "hybrid/executor/subgraph_executor.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/rts/rts_node_executor.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "graph/utils/graph_utils.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestSubgraphExecutor : public testing::Test {
 protected:
  void SetUp() {
    NodeExecutorManager::GetInstance().engine_mapping_.clear();
    auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
    engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
    engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);

    NodeExecutorManager::GetInstance().executors_.clear();
    auto &task_executor = NodeExecutorManager::GetInstance().executors_;
    task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new RtsNodeExecutor()));
    task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new GeLocalNodeExecutor()));
  }
  void TearDown() {
    NodeExecutorManager::GetInstance().engine_mapping_.clear();
    NodeExecutorManager::GetInstance().executors_.clear();
  }
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
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

static void CreateSimpleCondGraph(ComputeGraph &graph, NodePtr &switch_t, NodePtr &switch_f) {
/*******************************************************************************
 *             |
 *           Merge
 *          /     \.
 *  Active /       \ Active
 *        /         \.
 *       Add       Sub
 *      |   \     /   |
 *      |    \ _ /    |
 *      |    /   \    |
 *      |   /     \   |
 *    Switch       Switch
 *     |   \       /   |
 *     |    \     /    |
 *     |    Active     |
 *     |     \  /      |
 *     |     Less      |
 *     |     /   \     |
 *     |    /     \    |
 *      Data       Data
 ******************************************************************************/
  const auto data0 = CreateNode(graph, "data", DATA, 1, 1);
  const auto data1 = CreateNode(graph, "data1", DATA, 1, 1);
  data0->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  data1->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

  const auto const0 = CreateNode(graph, "const", CONSTANT, 0, 1);
  const auto const1 = CreateNode(graph, "const1", CONSTANT, 0, 1);
  const0->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  const1->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  {
    uint64_t const_value = 101;
    const auto op_desc = const0->GetOpDesc();
    auto weight = make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
    AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
  }
  {
    uint64_t const_value = 101;
    const auto op_desc = const1->GetOpDesc();
    auto weight = make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
    AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight);
  }

  const auto less1 = CreateNode(graph, "less", IDENTITY, 2, 1);  // Mock for less, just pass input0.

  const auto active1 = CreateNode(graph, "active1", STREAMACTIVE, 0, 0);
  switch_t = CreateNode(graph, "switch_t", STREAMSWITCH, 2, 0);
  switch_f = CreateNode(graph, "switch_f", STREAMSWITCH, 2, 0);
  AttrUtils::SetInt(switch_t->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND, RT_EQUAL); // 101 for true.
  AttrUtils::SetInt(switch_f->GetOpDesc(), ATTR_NAME_STREAM_SWITCH_COND, RT_NOT_EQUAL);

  const auto add1 = CreateNode(graph, "add", IDENTITY, 2, 1);  // Mock for add, just pass input0.
  const auto sub1 = CreateNode(graph, "sub", IDENTITY, 2, 1);  // Mock for sub, just pass input0.

  const auto merge1 = CreateNode(graph, "merge", STREAMMERGE, 2, 2);
  const auto active2 = CreateNode(graph, "active2", STREAMACTIVE, 0, 0);
  const auto active3 = CreateNode(graph, "active3", STREAMACTIVE, 0, 0);

  const auto iteration1 = CreateNode(graph, "iteration1", NEXTITERATION, 1, 1);
  const auto output1 = CreateNode(graph, "net_output", NETOUTPUT, 1, 1);
  output1->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(const0->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(const1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));

  GraphUtils::AddEdge(less1->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_t->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_f->GetInControlAnchor());

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), add1->GetInControlAnchor());
  GraphUtils::AddEdge(add1->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), sub1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), sub1->GetInDataAnchor(1));
  GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), sub1->GetInControlAnchor());
  GraphUtils::AddEdge(sub1->GetOutControlAnchor(), active3->GetInControlAnchor());
  GraphUtils::AddEdge(active3->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), iteration1->GetInDataAnchor(0));
  GraphUtils::AddEdge(iteration1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));
}

TEST_F(UtestSubgraphExecutor, simple_schedule_tasks) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  const auto data0 = CreateNode(*graph, "data", DATA, 1, 1);
  const auto output0 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);
  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), output0->GetInDataAnchor(0));
  data0->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");
  output0->GetOpDesc()->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);

  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.Build(), SUCCESS);

  uint64_t value_0 = 110;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  const std::vector<TensorValue> inputs{ in_tensor0 };

  uint64_t value_1 = 123;
  TensorValue out_tensor0(&value_1, sizeof(value_1));
  const std::vector<TensorValue> outputs{ out_tensor0 };

  auto input_desc = output0->GetOpDesc()->GetInputDescPtr(0);
  const std::vector<ConstGeTensorDescPtr> input_descs{ input_desc };

  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());

  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &graph_context);
  ASSERT_EQ(executor.ExecuteAsync(inputs, input_descs, outputs), SUCCESS);
  ASSERT_EQ(executor.Synchronize(), SUCCESS);
}

TEST_F(UtestSubgraphExecutor, cond_graph_schedule_tasks) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr switch_t = nullptr;
  NodePtr switch_f = nullptr;
  CreateSimpleCondGraph(*graph, switch_t, switch_f);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  std::vector<uint64_t> weights_value{101, 102};
  ge_sub_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weights_value.size() * sizeof(uint64_t)));
  ge_sub_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));

  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.Build(), SUCCESS);

  uint64_t value_0 = 101; // Enter used for Less, will pass this value to switch.
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  uint64_t value_1 = 110;
  TensorValue in_tensor1(&value_1, sizeof(value_1));
  const std::vector<TensorValue> inputs{ in_tensor0, in_tensor1 };
  uint64_t value_2 = 123;
  TensorValue out_tensor0(&value_2, sizeof(value_2));
  const std::vector<TensorValue> outputs{ out_tensor0 };

  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(*tensor_desc, 64);
  const std::vector<ConstGeTensorDescPtr> input_desc{ tensor_desc, tensor_desc };

  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  graph_context.allocator = NpuMemoryAllocator::GetAllocator(0);
  graph_context.callback_manager = std::unique_ptr<CallbackManager>(new CallbackManager());
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto root_graph = hybrid_model.root_graph_;
  switch_t = root_graph->FindNode("switch_t");
  switch_f = root_graph->FindNode("switch_f");
  const auto node_it_t = hybrid_model.node_items_.find(switch_t);
  const auto node_it_f = hybrid_model.node_items_.find(switch_f);
  ASSERT_NE(hybrid_model.node_items_.end(), node_it_t);
  ASSERT_NE(hybrid_model.node_items_.end(), node_it_f);

  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &graph_context);
  ASSERT_EQ(executor.ExecuteAsync(inputs, input_desc, outputs), SUCCESS);
  ASSERT_EQ(executor.Synchronize(), SUCCESS);

  const auto state_it_t = executor.subgraph_context_->node_states_.find(node_it_t->second.get());
  const auto state_it_f = executor.subgraph_context_->node_states_.find(node_it_f->second.get());
  ASSERT_NE(executor.subgraph_context_->node_states_.end(), state_it_t);
  ASSERT_NE(executor.subgraph_context_->node_states_.end(), state_it_f);
  ASSERT_EQ(state_it_t->second->GetSwitchIndex(), 1);
  ASSERT_EQ(state_it_f->second->GetSwitchIndex(), 0);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
}

TEST_F(UtestSubgraphExecutor, partial_execution_init) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ASSERT_NE(graph, nullptr);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  ASSERT_NE(ge_root_model, nullptr);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());
  hybrid_model.root_graph_item_->is_dynamic_ = false;
  GraphExecutionContext graph_context;
  SubgraphExecutor executor(hybrid_model.GetRootGraphItem(), &graph_context);

  ASSERT_EQ(executor.Init({}, {}), SUCCESS);
  ASSERT_EQ(executor.InitForPartialExecution({}, {}), SUCCESS);
}
} // namespace ge
