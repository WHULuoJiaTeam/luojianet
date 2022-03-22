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

#include "graph/passes/replace_with_empty_const_pass.h"

#include <gtest/gtest.h>
#include <set>
#include <string>

#include "graph_builder_utils.h"

namespace ge {
class UtestReplaceWithEmptyConstPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/// data1    const1
///      \ /
///     add1
///      |
///    cast1(empty)
///      |
///    conv2d
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 0);

  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape({1,1,8,8}),FORMAT_NCHW));
  GeTensorDesc empty_tensor(GeShape({1,0,8,8}),FORMAT_NCHW);
  cast1->GetOpDesc()->UpdateOutputDesc(0,empty_tensor);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, conv2d, 0);
  return builder;
}

///           data1    const1
///                \ /
///               add1
///                |
///   data2  -> switch1 (empty)
///                |
///              conv2d
ut::GraphBuilder Graph2Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph2");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 1);
  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 0);

  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  add1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape({1, 1, 8, 8}),FORMAT_NCHW));
  GeTensorDesc empty_tensor(GeShape({1, 0, 8, 8}),FORMAT_NCHW);
  switch1->GetOpDesc()->UpdateOutputDesc(0, empty_tensor);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, switch1, 0);
  builder.AddDataEdge(data2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, conv2d, 0);
  return builder;
}
}  // namespace


TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_const_success) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(),5);
  // run pass on add1, graph still has 5 nodes
  auto add1 = graph->FindNode("add1");
  Status ret = replace_with_empty_const_pass.Run(add1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(),5);

  // run pass on const1, graph still has 5 nodes
  auto const1 = graph->FindNode("const1");
  ret = replace_with_empty_const_pass.Run(const1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(),5);

  auto cast1 = graph->FindNode("cast1");
  ret = replace_with_empty_const_pass.Run(cast1);
  EXPECT_EQ(cast1->GetOutAllNodes().size(),0);
  auto conv2d = graph->FindNode("conv2d");
  EXPECT_EQ(conv2d->GetInDataNodes().at(0)->GetType(),"Const");
}

TEST_F(UtestReplaceWithEmptyConstPass, replace_whith_empty_switch_skip) {
  auto builder = Graph2Builder();
  auto graph = builder.GetGraph();
  graph->SetSessionID(0);
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;

  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
  // run pass on switch1, graph still has 6 nodes
  auto switch1 = graph->FindNode("switch1");
  EXPECT_NE(switch1, nullptr);
  Status ret = replace_with_empty_const_pass.Run(switch1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 6);
}
}  // namespace ge
