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
#include <memory>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_var_manager.h"

#define private public
#define protected public
#include "graph/preprocess/graph_preprocess.h"
#include "ge/ge_api.h"
#undef private
#undef protected

using namespace std;
namespace ge {
class UtestGraphPreproces : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

ComputeGraphPtr BuildGraph1(){
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1",DATA,1,1);
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);
  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({22, -1}));
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph5() {
  auto builder = ut::GraphBuilder("g5");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}

/*
 *   MapIndex   Data1          subgraph1        subgraph2
 *         \    /
 *          Case      ===>       Data2            Data3
 *           |
 *       Netoutput
 */
ComputeGraphPtr BuildGraph4() {
  auto builder = ut::GraphBuilder("mbatch_Case");

  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "case1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);

  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4_Subgraph(string graph_name) {
  auto builder = ut::GraphBuilder(graph_name);
  auto data1 = builder.AddNode(graph_name + "_data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph6() {
  auto builder = ut::GraphBuilder("g6");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {3, -1, -1, 5});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {});
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraphPreproces, test_dynamic_input_shape_parse) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph6();
  // prepare user_input & graph option
  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_FLOAT);
  GeTensor input1(tensor1);
  ge::GeTensorDesc tensor2;
  tensor2.SetFormat(ge::FORMAT_NCHW);
  tensor2.SetShape(ge::GeShape());
  tensor2.SetDataType(ge::DT_FLOAT);
  GeTensor input2(tensor2);
  std::vector<GeTensor> user_input = {input1, input2};
  std::map<string,string> graph_option = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[3,1~20,2~10,5],[]"}};
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::SUCCESS);
  // check data1 node output shape_range and shape
  auto data_node = graph_prepare.compute_graph_->FindNode("input1");
  auto data_output_desc = data_node->GetOpDesc()->GetOutputDescPtr(0);
  vector<int64_t> input1_expect_shape = {3,-1,-1,5};
  vector<std::pair<int64_t, int64_t>> intpu1_expect_shape_range = {{3,3},{1,20},{2,10},{5,5}};
  auto input1_result_shape = data_output_desc->GetShape();
  vector<std::pair<int64_t, int64_t>> input1_result_shape_range;
  data_output_desc->GetShapeRange(input1_result_shape_range);
  EXPECT_EQ(input1_result_shape.GetDimNum(), input1_expect_shape.size());
  EXPECT_EQ(input1_result_shape_range.size(), input1_expect_shape.size());
  for(size_t i =0; i< input1_expect_shape.size(); ++i){
      EXPECT_EQ(input1_result_shape.GetDim(i), input1_expect_shape.at(i));
  }
  for(size_t i =0; i< intpu1_expect_shape_range.size(); ++i){
     EXPECT_EQ(input1_result_shape_range.at(i).first, intpu1_expect_shape_range.at(i).first);
     EXPECT_EQ(input1_result_shape_range.at(i).second, intpu1_expect_shape_range.at(i).second);
  }
  // check data2 node output shape_range and shape
  auto data_node_2 = graph_prepare.compute_graph_->FindNode("input2");
  auto data_output_desc_2 = data_node_2->GetOpDesc()->GetOutputDescPtr(0);
  vector<std::pair<int64_t, int64_t>> intput2_result_shape_range;
  data_output_desc_2->GetShapeRange(intput2_result_shape_range);
  EXPECT_EQ(intput2_result_shape_range.size(), 0);
}

TEST_F(UtestGraphPreproces, test_update_input_fail) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph1();

  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_UNDEFINED);
  GeTensor input1(tensor1);
  std::vector<GeTensor> user_input = {input1};
  std::map<string,string> graph_option;
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGraphPreproces, test_check_user_input) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph1();

  vector<int64_t> dim = {2, -3};
  GeTensor tensor;
  tensor.SetTensorDesc(GeTensorDesc(GeShape(dim)));
  std::vector<GeTensor> user_input;
  user_input.emplace_back(tensor);

  Status ret = graph_prepare.CheckUserInput(user_input);
  EXPECT_EQ(ret, GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphPreproces, test_update_input_output1) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3();

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}


TEST_F(UtestGraphPreproces, check_ref_op_data_succ) {
  GraphPrepare graph_preparer;
  ComputeGraphPtr graph_test = BuildGraph5();
  NodePtr add_node = nullptr;
  for (auto &node : graph_test->GetAllNodes()) {
    if (node->GetName() == "add") {
      add_node = node;
    }
  }
  EXPECT_NE(add_node, nullptr);
  string input_name = "__input0";
  std::set<NodePtr> ref_nodes;
  auto ret = graph_preparer.CheckRefInputNode(add_node, input_name, ref_nodes);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_update_dtype_mbatch_case) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph4();
  auto parent_graph = graph_prepare.compute_graph_;
  auto subgraph1 = BuildGraph4_Subgraph("subgraph1");
  auto subgraph2 = BuildGraph4_Subgraph("subgraph2");

  auto data1 = parent_graph->FindNode("data1");
  auto data_desc = data1->GetOpDesc();

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  case_node->GetOpDesc()->AddSubgraphName("subgraph1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph1->SetParentNode(case_node);
  subgraph1->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph1", subgraph1), GRAPH_SUCCESS);

  case_node->GetOpDesc()->AddSubgraphName("subgraph2");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "subgraph2");
  subgraph2->SetParentNode(case_node);
  subgraph2->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph2", subgraph2), GRAPH_SUCCESS);

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);

  auto case_desc = case_node->GetOpDesc();
  auto case_input = case_desc->MutableInputDesc(1);
  EXPECT_EQ(case_input->GetDataType(), 1);

  auto sub1_data1 = subgraph1->FindNode("subgraph1_data1");
  EXPECT_NE(sub1_data1, nullptr);
  auto data1_desc = sub1_data1->GetOpDesc();
  auto data1_input = data1_desc->MutableInputDesc(0);
  EXPECT_EQ(data1_input->GetDataType(), 1);
  auto data1_output = data1_desc->MutableOutputDesc(0);
  EXPECT_EQ(data1_output->GetDataType(), 1);
}

TEST_F(UtestGraphPreproces, test_prepare_dyn_shape) {
  ComputeGraphPtr compute_graph = BuildGraph5();
  GraphPtr graph_ptr = std::make_shared<Graph>(GraphUtils::CreateGraphFromComputeGraph(compute_graph));

  GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetGraph(graph_ptr);

  std::vector<GeTensor> user_input;
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.PrepareDynShape(graph_node, user_input, compute_graph, 0), SUCCESS);
}

TEST_F(UtestGraphPreproces, test_updar_variable_formats) {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1);
  auto g1 = builder.GetGraph();
  g1->SetSessionID(0);
  TransNodeInfo trans_node_info;
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), INTERNAL_ERROR);

  auto builder1 = ut::GraphBuilder("g2");
  auto var1 = builder1.AddNode("var1", VARIABLE, 1, 1);
  auto g2 = builder1.GetGraph();
  g2->SetSessionID(0);
  VarTransRoad fusion_road1;
  VarManager::Instance(g2->GetSessionID())->SetTransRoad(var1->GetName(), fusion_road1);
  AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "var1");
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g2), SUCCESS);
}
}