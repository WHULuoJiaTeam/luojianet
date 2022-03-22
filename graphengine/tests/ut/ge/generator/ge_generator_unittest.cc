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

#define private public
#define protected public
#include "generator/ge_generator.h"
#include "graph/utils/tensor_utils.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_impl.h"
#include "../graph/passes/graph_builder_utils.h"
#include "../graph/manager/graph_manager.h"
#include "all_ops.h"

using namespace std;

namespace ge {
class UtestGeGenerator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
ComputeGraphPtr MakeGraph() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  builder.AddDataEdge(data, 0, addn1, 0);
  return builder.GetGraph();
}

static GeAttrValue::NamedAttrs CreateNamedAttrs(const string &name, std::map<string, GeAttrValue> map) {
  GeAttrValue::NamedAttrs named_attrs;
  named_attrs.SetName(name);
  for (auto it : map) {
    named_attrs.SetAttr(it.first, it.second);
  }
  return named_attrs;
}
}  // namespace

/*
TEST_F(UtestGeGenerator, test_build_single_op_offline) {
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 512);

  shared_ptr<OpDesc> op_desc = make_shared<OpDesc>("Add", "add");
  EXPECT_EQ(op_desc->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  // not Initialize, impl is null.
  GeGenerator generator;
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs, outputs, "offline_"), PARAM_INVALID);

  // const map<string, string> &options
  generator.Initialize({});
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs, outputs, "offline_"), GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED);
}
*/
graphStatus TestFunc(Operator &op) { return 0; }
graphStatus TestFunc1(Operator &op) { return 1; }
TEST_F(UtestGeGenerator, test_infer_format_for_single_op) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("graph_name"); 
  auto graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  OperatorFactoryImpl::RegisterInferFormatFunc("Add", TestFunc);
  shared_ptr<OpDesc> op_desc = make_shared<OpDesc>("add", "add");
  compute_graph->AddNode(op_desc);
  GeGenerator generator;
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc, graph), SUCCESS);
  shared_ptr<OpDesc> op_desc1 = make_shared<OpDesc>("Add", "Add");
  compute_graph->AddNode(op_desc1);
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc1, graph), SUCCESS);
  OperatorFactoryImpl::RegisterInferFormatFunc("MatMulV2", TestFunc1);
  shared_ptr<OpDesc> op_desc2 = make_shared<OpDesc>("MatMulV2", "MatMulV2");
  GeTensorDesc tensor_desc;
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);
  compute_graph->AddNode(op_desc2);
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc2, graph), FAILED);
}

TEST_F(UtestGeGenerator, test_build_single_op_online) {
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_AIVECTOR, false, model_buffer), FAILED);
}

TEST_F(UtestGeGenerator, test_check_aicore) {
  GeGenerator generator;
  generator.Initialize({});
  auto graph = MakeGraph();
  EXPECT_EQ(generator.CheckNoAicore(graph), true);
}

TEST_F(UtestGeGenerator, test_graph_manager) {
  GraphManager graph_manager;
  GraphPartitioner graph_partitioner;

  auto root_graph = MakeGraph();
  auto sub_graph = MakeGraph();
  root_graph->AddSubGraph(sub_graph);

  auto sgi = MakeShared<SubGraphInfo>();
  // set engine name
  sgi->SetEngineName("AIcoreEngine");
  sgi->SetSubGraph(sub_graph);

  auto sgi_gelocal = MakeShared<SubGraphInfo>();
  // set engine name
  sgi_gelocal->SetEngineName("GELOCAL");
  sgi_gelocal->SetSubGraph(sub_graph);

  graph_partitioner.graph_2_input_subgraph_[root_graph] = sgi_gelocal;
  graph_partitioner.graph_2_subgraph_list_.insert({root_graph, {sgi, sgi_gelocal}});
  graph_partitioner.graph_2_subgraph_list_.insert({sub_graph, {sgi, sgi_gelocal}});
  EXPECT_EQ(graph_manager.ConvertGraphToFile(root_graph, graph_partitioner, "./"), GRAPH_SUCCESS);
}

TEST_F(UtestGeGenerator, test_set_model_name) {
  GeGenerator generator;
  generator.Initialize({});
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(GeRootModel());
  ComputeGraphPtr graph = make_shared<ComputeGraph>(ComputeGraph("graph"));
  (void)AttrUtils::SetBool(graph, "_dynamic_shape_partitioned", true);
  ge_root_model->root_graph_ = std::move(graph);
  EXPECT_EQ(generator.SetModelNameForDump(ge_root_model), SUCCESS);
}

TEST_F(UtestGeGenerator, test_remove_const) {
  GeGenerator generator;
  GeTensorDesc tensor_desc;
  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = {tensor};
  vector<GeTensor> outputs;
  generator.RemoveConst(inputs, outputs);
}

TEST_F(UtestGeGenerator, test_generate_online_model) {
  GeTensorDesc tensor_desc;
  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  GeGenerator generator;
  generator.Initialize({});
  std::string name;
  EXPECT_NE(generator.GenerateOfflineModel(graph, name, inputs), SUCCESS);
}
}  // namespace ge
