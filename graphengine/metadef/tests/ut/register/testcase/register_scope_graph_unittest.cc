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
#include <iostream>
#include "graph/debug/ge_attr_define.h"
#define private public
#define protected public
#include "external/register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#undef private
#undef protected

using namespace ge;
class UtestScopeGraph : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

///        placeholder0  placeholder1
///          |       /\  /\       |
///          |      /  \/  \      |
///          |     /   /\   \     |
///          |     |  /  \  |     |
///          |     add0   mul0    |
///          |     /     /c | \   |
///            mul1 --- /   |   add1
///              \          |    |
///               \ ---- add2    |
///                      |       |
///                    retval0 retval1

void CreateGraphDef(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}


TEST_F(UtestScopeGraph, test_build_scope_graph_succ) {
  domi::tensorflow::GraphDef graph_def;

  CreateGraphDef(graph_def);
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);
  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 9);

  // checkpoint 1
  auto mul0_iter = nodes_map.find("mul0");
  ASSERT_NE(mul0_iter, nodes_map.end());
  std::vector<std::string> mul0_inputs;
  std::vector<std::string> mul0_outputs;
  mul0_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, mul0_inputs);
  mul0_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, mul0_outputs);
  ASSERT_EQ(mul0_inputs.size(), 2);
  EXPECT_EQ(mul0_inputs.at(0), "0:placeholder0:0");
  EXPECT_EQ(mul0_inputs.at(1), "1:placeholder1:0");
  ASSERT_EQ(mul0_outputs.size(), 3);
  EXPECT_EQ(mul0_outputs.at(0), "-1:mul1:-1");
  EXPECT_EQ(mul0_outputs.at(1), "0:add1:0");
  EXPECT_EQ(mul0_outputs.at(2), "0:add2:1");

  // checkpoint 2
  auto mul1_iter = nodes_map.find("mul1");
  ASSERT_NE(mul1_iter, nodes_map.end());
  std::vector<std::string> mul1_inputs;
  std::vector<std::string> mul1_outputs;
  mul1_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, mul1_inputs);
  mul1_iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, mul1_outputs);
  ASSERT_EQ(mul1_inputs.size(), 3);
  EXPECT_EQ(mul1_inputs.at(0), "-1:mul0:-1");
  EXPECT_EQ(mul1_inputs.at(1), "0:placeholder0:0");
  EXPECT_EQ(mul1_inputs.at(2), "1:add0:0");
  ASSERT_EQ(mul1_outputs.size(), 1);
  EXPECT_EQ(mul1_outputs.at(0), "0:add2:0");
}

TEST_F(UtestScopeGraph, test_build_scope_graph_node_without_inout) {
  domi::tensorflow::GraphDef graph_def;
  auto no_op = graph_def.add_node();
  no_op->set_name("no_op");
  no_op->set_op("NoOp");

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;
  impl->BuildScopeGraph(&graph_def);

  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 1);
  auto iter = nodes_map.find("no_op");
  ASSERT_NE(iter, nodes_map.end());
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  graphStatus get_input_attr = iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS, inputs);
  graphStatus get_output_attr = iter->second->GetAttr(ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS, outputs);
  ASSERT_EQ(get_input_attr, GRAPH_SUCCESS);
  ASSERT_EQ(get_output_attr, GRAPH_SUCCESS);
  EXPECT_EQ(inputs.size(), 0);
  EXPECT_EQ(outputs.size(), 0);
}

TEST_F(UtestScopeGraph, test_build_scope_graph_failed) {
  domi::tensorflow::GraphDef graph_def;
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();

  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  ASSERT_NE(scope_graph, nullptr);
  Status ret = scope_graph->Init();
  ASSERT_EQ(ret, SUCCESS);
  auto &impl = scope_graph->impl_;

  // 1. input name is invalied
  add0->set_input(0, "placeholder0:invalid:input");
  impl->BuildScopeGraph(&graph_def);
  auto nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // 2. index is invalid
  add0->set_input(0, "placeholder0:s1");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // 3. index is out of range
  add0->set_input(0, "placeholder0:12356890666666");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);

  // index is negative
  add0->set_input(0, "placeholder0:-1");
  impl->BuildScopeGraph(&graph_def);
  nodes_map = impl->GetNodesMap();
  EXPECT_EQ(nodes_map.size(), 0);
}
