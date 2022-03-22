/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "graph/operator_reg.h"
#include "graph/utils/graph_utils.h"
#include "graph/attr_value.h"

namespace ge {
REG_OP(Const)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const);

REG_OP(OCG2)
    .DYNAMIC_INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(OCG2);

REG_OP(OCG3)
    .INPUT(x,
           TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                       DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(OCG3);

REG_OP(OCG4)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(OCG4);

REG_OP(OCG5)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(OCG5);

class OperatorConstructGraphUt : public testing::Test {};

/**
 *            c
 *      ocg2 ----> const
 *       |
 *      ocg2
 *        |
 *      ocg3
 *    /    \
 * const   const
 */
Graph BuildGraph1() {
  auto o1_1 = op::Const("o1_1");
  auto o1_2 = op::Const("o1_2");
  auto o3 = op::OCG3("o3");
  auto o2_1 = op::OCG2("o2_1");
  auto o2_2 = op::OCG2("o2_2");
  auto o1_3 = op::Const("o1_3");

  TensorDesc td{Shape(std::vector<int64_t>({8, 3, 224, 224})), FORMAT_NCHW, DT_UINT8};
  Tensor tensor(td);
  tensor.SetData(std::vector<uint8_t>(8 * 3 * 224 * 224));

  o1_1.set_attr_value(tensor);
  o1_2.set_attr_value(tensor);
  o1_3.set_attr_value(tensor);
  o3.set_input_x(o1_1).set_input_shape_by_name(o1_2, "y");
  o2_1.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o3);
  o2_2.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o2_1, "y");
  o1_3.AddControlInput(o2_2);

  Graph g{"name"};
  g.SetInputs(std::vector<Operator>({o1_1, o1_2})).SetOutputs(std::vector<Operator>({o2_2, o1_3}));
  return g;
}
Graph BuildGraph1ByIndex() {
  auto o1_1 = op::Const("o1_1");
  auto o1_2 = op::Const("o1_2");
  auto o3 = op::OCG3("o3");
  auto o2_1 = op::OCG2("o2_1");
  auto o2_2 = op::OCG2("o2_2");
  auto o1_3 = op::Const("o1_3");

  TensorDesc td{Shape(std::vector<int64_t>({8, 3, 224, 224})), FORMAT_NCHW, DT_UINT8};
  Tensor tensor(td);
  tensor.SetData(std::vector<uint8_t>(8 * 3 * 224 * 224));

  o1_1.set_attr_value(tensor);
  o1_2.set_attr_value(tensor);
  o1_3.set_attr_value(tensor);
  o3.set_input_x(o1_1, 0).set_input_shape(o1_2, 0);
  o2_1.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o3);
  o2_2.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o2_1, "y");
  o1_3.AddControlInput(o2_2);

  Graph g{"name"};
  g.SetInputs(std::vector<Operator>({o1_1, o1_2})).SetOutputs(std::vector<Operator>({o2_2, o1_3}));
  return g;
}

void CheckGraph1(Graph &g) {
  auto cg = GraphUtils::GetComputeGraph(g);
  EXPECT_NE(cg, nullptr);

  EXPECT_EQ(cg->GetAllNodesSize(), 6);
  auto node_o1_1 = cg->FindNode("o1_1");
  auto node_o1_2 = cg->FindNode("o1_2");
  auto node_o1_3 = cg->FindNode("o1_3");
  auto node_o2_1 = cg->FindNode("o2_1");
  auto node_o2_2 = cg->FindNode("o2_2");
  auto node_o3 = cg->FindNode("o3");
  EXPECT_NE(node_o1_1, nullptr);
  EXPECT_NE(node_o1_2, nullptr);
  EXPECT_NE(node_o1_3, nullptr);
  EXPECT_NE(node_o2_1, nullptr);
  EXPECT_NE(node_o2_2, nullptr);
  EXPECT_NE(node_o3, nullptr);

  EXPECT_EQ(node_o1_1->GetName(), "o1_1");
  EXPECT_EQ(node_o1_2->GetName(), "o1_2");
  EXPECT_EQ(node_o1_3->GetName(), "o1_3");
  EXPECT_EQ(node_o2_1->GetName(), "o2_1");
  EXPECT_EQ(node_o2_2->GetName(), "o2_2");
  EXPECT_EQ(node_o3->GetName(), "o3");

  EXPECT_EQ(node_o1_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node_o1_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx(), 0);
  EXPECT_EQ(node_o1_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "o3");

  EXPECT_EQ(node_o1_2->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node_o1_2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx(), 1);
  EXPECT_EQ(node_o1_2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "o3");

  EXPECT_EQ(node_o3->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node_o3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx(), 0);
  EXPECT_EQ(node_o3->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "o2_1");

  EXPECT_EQ(node_o2_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node_o2_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx(), 0);
  EXPECT_EQ(node_o2_1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "o2_2");

  EXPECT_EQ(node_o2_2->GetOutControlNodes().size(), 1);
  EXPECT_EQ(node_o2_2->GetOutControlNodes().at(0)->GetName(), "o1_3");
}

TEST_F(OperatorConstructGraphUt, ConstructGraph1) {
  auto g = BuildGraph1();
  CheckGraph1(g);
}

TEST_F(OperatorConstructGraphUt, ConstructWithIndex) {
  auto g = BuildGraph1ByIndex();
  CheckGraph1(g);
}

TEST_F(OperatorConstructGraphUt, GetInputConstData1) {
  auto o1_1 = op::Const("o1_1");
  auto o1_2 = op::Const("o1_2");
  auto o3 = op::OCG3("o3");
  auto o2_1 = op::OCG2("o2_1");
  auto o2_2 = op::OCG2("o2_2");
  auto o1_3 = op::Const("o1_3");

  TensorDesc td{Shape(std::vector<int64_t>({8, 3, 224, 224})), FORMAT_NCHW, DT_UINT8};
  Tensor tensor(td);
  tensor.SetData(std::vector<uint8_t>(8 * 3 * 224 * 224));

  o1_1.set_attr_value(tensor);
  o1_2.set_attr_value(tensor);
  o1_3.set_attr_value(tensor);
  o3.set_input_x(o1_1).set_input_shape_by_name(o1_2, "y");
  o2_1.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o3);
  o2_2.create_dynamic_input_x(1, true).set_dynamic_input_x(0, o2_1, "y");
  o1_3.AddControlInput(o2_2);

  Graph g{"name"};
  g.SetInputs(std::vector<Operator>({o1_1, o1_2})).SetOutputs(std::vector<Operator>({o2_2, o1_3}));

  Tensor t1;
  EXPECT_NE(o2_1.GetInputConstData("x1", t1), GRAPH_SUCCESS);
  EXPECT_EQ(o3.GetInputConstData("x", t1), GRAPH_SUCCESS);
  EXPECT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
}

TEST_F(OperatorConstructGraphUt, SetGetAttrOk) {
  auto op = OperatorFactory::CreateOperator("op", "OCG3");
  int64_t value = 10;
  EXPECT_NE(op.GetAttr("Hello", value), GRAPH_SUCCESS);
  op.SetAttr("Hello", 10);
  EXPECT_EQ(op.GetAttr("Hello", value), GRAPH_SUCCESS);
  EXPECT_EQ(value, 10);
}

TEST_F(OperatorConstructGraphUt, SetGetAttrByAnyValueOk) {
  auto op = OperatorFactory::CreateOperator("op", "OCG3");
  int64_t value = 10;
  op.SetAttr("Foo", AttrValue::CreateFrom<int64_t>(10));
  EXPECT_EQ(op.GetAttr("Foo", value), GRAPH_SUCCESS);
  EXPECT_EQ(value, 10);

  AttrValue attr_value;
  EXPECT_NE(op.GetAttr("Bar", attr_value), GRAPH_SUCCESS);
  EXPECT_EQ(op.GetAttr("Foo", attr_value), GRAPH_SUCCESS);
  value = 0;
  attr_value.GetValue<int64_t>(value);
  EXPECT_EQ(value, 10);
}

TEST_F(OperatorConstructGraphUt, GetIntputOutputSizeOk) {
  auto op = OperatorFactory::CreateOperator("op", "OCG3");
  EXPECT_EQ(op.GetInputsSize(), 2);
  EXPECT_EQ(op.GetOutputsSize(), 1);
}

TEST_F(OperatorConstructGraphUt, UpdateInputOutputOk) {
  TensorDesc td;
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NHWC);
  td.SetShape(Shape(std::vector<int64_t>({8, 1, 224, 224, 16})));
  td.SetOriginShape(Shape(std::vector<int64_t>({8, 224, 224, 3})));

  auto op = OperatorFactory::CreateOperator("op", "OCG3");
  EXPECT_EQ(op.UpdateInputDesc("x", td), GRAPH_SUCCESS);
  EXPECT_EQ(op.UpdateInputDesc("shape", td), GRAPH_SUCCESS);
  EXPECT_EQ(op.UpdateOutputDesc("y", td), GRAPH_SUCCESS);
  EXPECT_NE(op.UpdateInputDesc("xx", td), GRAPH_SUCCESS);
  EXPECT_NE(op.UpdateOutputDesc("yy", td), GRAPH_SUCCESS);

  EXPECT_EQ(op.GetInputDesc(0).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(op.GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(op.GetInputDescByName("x").GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(op.GetInputDescByName("x").GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(op.GetOutputDesc(0).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(op.GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(op.GetOutputDescByName("y").GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(op.GetOutputDescByName("y").GetOriginFormat(), FORMAT_NHWC);
}

TEST_F(OperatorConstructGraphUt, UpdateOutputOk_AutoUpdatedToPeer) {
  auto o1_1 = op::Const("o1_1");
  auto o1_2 = op::Const("o1_2");
  auto o3 = op::OCG3("o3");

  TensorDesc td{Shape(std::vector<int64_t>({8, 3, 224, 224})), FORMAT_NCHW, DT_UINT8};
  Tensor tensor(td);
  tensor.SetData(std::vector<uint8_t>(8 * 3 * 224 * 224));

  o1_1.set_attr_value(tensor);
  o1_2.set_attr_value(tensor);
  o3.set_input_x(o1_1).set_input_shape_by_name(o1_2, "y");

  Graph g{"name"};
  g.SetInputs(std::vector<Operator>({o1_1, o1_2})).SetOutputs(std::vector<Operator>({o3}));

  TensorDesc td1;
  td1.SetFormat(FORMAT_NC1HWC0);
  td1.SetOriginFormat(FORMAT_NHWC);
  td1.SetShape(Shape(std::vector<int64_t>({8, 1, 224, 224, 16})));
  td1.SetOriginShape(Shape(std::vector<int64_t>({8, 224, 224, 3})));

  o1_1.UpdateOutputDesc("y", td1);
  o1_2.UpdateOutputDesc("y", td1);

  EXPECT_EQ(o1_1.GetOutputDesc(0).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(o1_1.GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(o1_2.GetOutputDesc(0).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(o1_2.GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(o3.GetInputDesc(0).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(o3.GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
}

TEST_F(OperatorConstructGraphUt, SubgraphIrDefOk) {
  auto if_op = op::OCG4("if");
  std::vector<AscendString> names;
  EXPECT_EQ(if_op.GetSubgraphNamesCount(), 2);
  // 此接口获取的names是无序的
  EXPECT_EQ(if_op.GetSubgraphNames(names), GRAPH_SUCCESS);
  std::set<std::string> names_set;
  for (const auto &name : names) {
    names_set.insert(name.GetString());
  }
  EXPECT_EQ(names_set, std::set<std::string>({"then_branch", "else_branch"}));

  auto case_op = op::OCG5("case");
  EXPECT_EQ(case_op.GetSubgraphNamesCount(), 1);
  names.clear();
  EXPECT_EQ(case_op.GetSubgraphNames(names), GRAPH_SUCCESS);
  EXPECT_EQ(names.size(), 1);
  EXPECT_EQ(strcmp(names[0].GetString(), "branches"), 0);
}

TEST_F(OperatorConstructGraphUt, SetGetSubgraphBuilderOk1) {
  auto if_op = op::OCG4("if");
  if_op.set_subgraph_builder_else_branch([]() { return Graph("FromSubgraphBuilder1"); });
  if_op.set_subgraph_builder_then_branch([]() { return Graph("FromSubgraphBuilder2"); });

  auto else_graph = if_op.get_subgraph_builder_else_branch()();
  AscendString as;
  EXPECT_EQ(else_graph.GetName(as), GRAPH_SUCCESS);
  EXPECT_EQ(strcmp(as.GetString(), "FromSubgraphBuilder1"), 0);

  auto then_graph = if_op.get_subgraph_builder_then_branch()();
  EXPECT_EQ(then_graph.GetName(as), GRAPH_SUCCESS);
  EXPECT_EQ(strcmp(as.GetString(), "FromSubgraphBuilder2"), 0);

  EXPECT_EQ(if_op.GetSubgraphBuilder("Hello"), nullptr);
}

TEST_F(OperatorConstructGraphUt, SetGetSubgraphBuilderOk2) {
  auto case_op = op::OCG5("case");
  case_op.create_dynamic_subgraph_branches(3);
  case_op.set_dynamic_subgraph_builder_branches(0, []() { return Graph("case1"); });
  case_op.set_dynamic_subgraph_builder_branches(1, []() { return Graph("case2"); });
  case_op.set_dynamic_subgraph_builder_branches(2, []() { return Graph("case3"); });

  auto case1 = case_op.get_dynamic_subgraph_builder_branches(0)();
  AscendString as;
  EXPECT_EQ(case1.GetName(as), GRAPH_SUCCESS);
  EXPECT_EQ(strcmp(as.GetString(), "case1"), 0);

  auto case2 = case_op.get_dynamic_subgraph_builder_branches(1)();
  EXPECT_EQ(case2.GetName(as), GRAPH_SUCCESS);
  EXPECT_EQ(strcmp(as.GetString(), "case2"), 0);

  auto case3 = case_op.get_dynamic_subgraph_builder_branches(2)();
  EXPECT_EQ(case3.GetName(as), GRAPH_SUCCESS);
  EXPECT_EQ(strcmp(as.GetString(), "case3"), 0);
}
}  // namespace ge