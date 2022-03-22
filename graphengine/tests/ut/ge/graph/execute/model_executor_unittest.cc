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

#define protected public
#define private public
#include "graph/execute/model_executor.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"

using namespace std;

namespace ge {
class UtestModelExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
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

TEST_F(UtestModelExecutorTest, test_load_graph_sync) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(false);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_load_graph_async) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_load_graph_failed) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // GeModel is null, DavinciModel::Assign will return FAILED
  setenv(kEnvGeuseStaticMemory, "1", true);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), FAILED);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  unsetenv(kEnvGeuseStaticMemory);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_memory) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    ModelManager::GetInstance()->InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model1->SetId(2);
    ModelManager::GetInstance()->InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  GeModelPtr ge_model = make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, parse_inputs_dims_data) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  OmeContext context;
  SetLocalOmeContext(context);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  const auto data1 = CreateNode(*compute_graph, DATA, "data1", 1, 1);
  const auto next1 = CreateNode(*compute_graph, GETNEXT, "data1", 1, 1);

  Tensor tensor;
  std::vector<Tensor> input_tensors;
  input_tensors.emplace_back(tensor);
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // dynamic_node_type is empty, just return

  context.dynamic_node_type = DATA;
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // ParseInputsDimsForData

  context.getnext_nosink_nodes.emplace_back(next1);
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // ParseInputsDimsForGetNexNosinkAndData

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, parse_inputs_dims_getnext) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  OmeContext context;
  SetLocalOmeContext(context);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  const auto data1 = CreateNode(*compute_graph, DATA, "data1", 1, 1);
  const auto next1 = CreateNode(*compute_graph, GETNEXT, "data1", 1, 1);

  Tensor tensor;
  std::vector<Tensor> input_tensors;
  input_tensors.emplace_back(tensor);

  context.dynamic_node_type = GETNEXT;
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // just getnext_sink

  context.getnext_nosink_nodes.emplace_back(next1);
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // ParseInputsDimsForData

  context.data_nodes.emplace_back(data1);
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), PARAM_INVALID);  // ParseInputsDimsForGetNexNosinkAndData
  AttrUtils::SetInt(next1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  EXPECT_EQ(model_executor.ParseInputsDims(input_tensors), SUCCESS);  // ParseInputsDimsForGetNexNosinkAndData

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_thread) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::Context error_context;
  GEThreadLocalContext context;
  const auto callback = [](Status status, std::vector<ge::Tensor> &outputs) { };

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);
  graph_node->IncreaseLoadCount();
  graph_node->Lock();

  Tensor tensor;
  std::vector<Tensor> input_tensors;
  input_tensors.emplace_back(tensor);

  RunArgs run_args{graph_node, graph_id, session_id, error_context, input_tensors, ge_root_model, context, callback};
  EXPECT_EQ(model_executor.PushGraph(run_args), SUCCESS);

  while (model_executor.run_args_q_.Size() > 0) {
    usleep(10);  // 0.01ms, Wait for RunThread.
  }
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

static void test_run_graph(ModelExecutor &model_executor) {
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(false);  // RunGraph is Synchronization.
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<GeTensor> inputs;
  std::vector<GeTensor> outputs;
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_train) {
  GetThreadLocalContext().SetGlobalOption({{OPTION_GRAPH_RUN_MODE, "1"}});
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  test_run_graph(model_executor);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_infer) {
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  test_run_graph(model_executor);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_with_stream) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  GraphId graph_id = 1;
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);

  GeTensor tensor;
  std::vector<GeTensor> inputs{tensor};
  std::vector<GeTensor> outputs;

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph_id, stream, inputs, outputs), 145003);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  rtStreamDestroy(stream);
}
} // namespace ge
