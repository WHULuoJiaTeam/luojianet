/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#define private public
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "inc/external/graph/operator_reg.h"
#include "inc/external/graph/operator.h"
#include "inc/external/graph/operator_factory.h"
#include "inc/graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/ge_attr_value.h"
#undef private

using namespace ge;
class UtestGraph : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  return graph;
}

TEST_F(UtestGraph, copy_graph_01) {
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");
  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);

  auto cp_compute_graph = ge::GraphUtils::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node = cp_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node, nullptr);
  ASSERT_NE(cp_add_node, add_node);
}


TEST_F(UtestGraph, copy_graph_02) {
  ge::OpDescPtr if_op(new ge::OpDesc("if", "If"));
  if_op->AddDynamicInputDesc("input", 1);
  if_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto if_node = compute_graph->AddNode(if_op);
  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");

  if_op->AddSubgraphName("then_branch");
  if_op->AddSubgraphName("else_branch");
  if_op->SetSubgraphInstanceName(0, "then");
  if_op->SetSubgraphInstanceName(1, "else");

  ge::OpDescPtr add_op1(new ge::OpDesc("add1", "Add"));
  add_op1->AddDynamicInputDesc("input", 2);
  add_op1->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> then_compute_graph(new ge::ComputeGraph("then"));
  auto add_node1 = then_compute_graph->AddNode(add_op1);
  then_compute_graph->SetParentNode(if_node);
  then_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(then_compute_graph);

  ge::OpDescPtr add_op2(new ge::OpDesc("add2", "Add"));
  add_op2->AddDynamicInputDesc("input", 2);
  add_op2->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> else_compute_graph(new ge::ComputeGraph("else"));
  auto add_node2 = else_compute_graph->AddNode(add_op2);
  else_compute_graph->SetParentNode(if_node);
  else_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(else_compute_graph);

  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);

  auto cp_compute_graph = ge::GraphUtils::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 1);
  auto cp_if_node = cp_compute_graph->FindNode("if");
  ASSERT_NE(cp_if_node, nullptr);
  ASSERT_NE(cp_if_node, if_node);

  auto cp_then_compute_graph = cp_compute_graph->GetSubgraph("then");
  ASSERT_NE(cp_then_compute_graph, nullptr);
  ASSERT_NE(cp_then_compute_graph, then_compute_graph);
  ASSERT_EQ(cp_then_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node1 = cp_then_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node1, nullptr);
  ASSERT_NE(cp_add_node1, add_node1);

  auto cp_else_compute_graph = cp_compute_graph->GetSubgraph("else");
  ASSERT_NE(cp_else_compute_graph, nullptr);
  ASSERT_NE(cp_else_compute_graph, else_compute_graph);
  ASSERT_EQ(cp_else_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node2 = cp_else_compute_graph->FindNode("add2");
  ASSERT_NE(cp_add_node2, nullptr);
  ASSERT_NE(cp_add_node2, add_node2);
}

REG_OP(Mul)
    .OP_END_FACTORY_REG(Mul)
IMPL_INFER_VALUE_RANGE_FUNC(Mul, func){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

REG_OP(Test2)
    .OP_END_FACTORY_REG(Test2)
IMPL_INFER_VALUE_RANGE_FUNC(Test2, func2){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_infer_value_range_register_succ) {
  string op_type = "Add";
  INFER_VALUE_RANGE_DEFAULT_REG(Add);
  INFER_VALUE_RANGE_DEFAULT_REG(Test1);
  auto para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_EQ(para.infer_value_func, nullptr);

  op_type = "Mul";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Mul, INPUT_HAS_VALUE_RANGE, func);
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Test2, INPUT_IS_DYNAMIC, func2);
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_NE(para.infer_value_func, nullptr);

  op_type = "Sub";
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, false);
}

REG_OP(Shape)
    .OP_END_FACTORY_REG(Shape)
IMPL_INFER_VALUE_RANGE_FUNC(Shape, ShapeValueInfer){
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_value_range_infer_and_set_get) {
  using std::make_pair;
  string op_type = "Shape";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueInfer);
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto shape_op_desc = std::make_shared<OpDesc>("node_name", op_type);
  GeTensorDesc tensor_desc(GeShape({-1, -1, 4, 192}), ge::FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240),
                                                          make_pair(4, 4),   make_pair(192, 192)};
  tensor_desc.SetShapeRange(shape_range);
  shape_op_desc->AddInputDesc(tensor_desc);
  GeTensorDesc out_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, DT_INT32);
  shape_op_desc->AddOutputDesc(out_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(shape_node);
  auto ret = shape_node->GetOpDesc()->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto output_0_desc = shape_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> value_range;
  output_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.size(), 4);

  std::vector<int64_t> target_value_range = {1, 100, 1, 240, 4, 4, 192, 192};
  std::vector<int64_t> output_value_range;
  for (auto pair : value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);
}

TEST_F(UtestGraph, get_all_graph_nodes) {
  ComputeGraphPtr graph = BuildComputeGraph();
  auto nodes = graph->GetAllNodes();
  EXPECT_EQ(nodes.size(), 5);
}
