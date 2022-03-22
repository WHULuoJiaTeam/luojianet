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

#include "graph/passes/reshape_remove_pass.h"

#include <gtest/gtest.h>
#include <set>
#include <string>

#include "graph_builder_utils.h"

namespace ge {
class UtestReshapeRemovePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/// netoutput1
///    |
/// transdata1
///    |
///  reshape1
///    |     \.
///   var1   const1
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  ;
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto reshape1 = builder.AddNode("reshape1", "Reshape", 2, 1);
  auto transdata1 = builder.AddNode("transdata1", "Transdata", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 1, 0);

  builder.AddDataEdge(var1, 0, reshape1, 0);
  builder.AddDataEdge(const1, 0, reshape1, 1);
  builder.AddDataEdge(reshape1, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);

  return builder;
}

///    netoutput1
///     |       \.
///transdata1    \.
///    |          \.
///  reshape1    reshape2
///    |     \    /    \.
///   var1   const1   var2
ut::GraphBuilder Graph2Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto reshape1 = builder.AddNode("reshape1", "Reshape", 2, 1);
  auto reshape2 = builder.AddNode("reshape2", "Reshape", 2, 1);
  auto transdata1 = builder.AddNode("transdata1", "Transdata", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 2, 0);

  builder.AddDataEdge(var1, 0, reshape1, 0);
  builder.AddDataEdge(const1, 0, reshape1, 1);
  builder.AddDataEdge(var2, 0, reshape2, 0);
  builder.AddDataEdge(const1, 0, reshape2, 1);
  builder.AddDataEdge(reshape1, 0, transdata1, 0);
  builder.AddDataEdge(reshape2, 0, netoutput1, 1);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);

  return builder;
}

///    netoutput1
///     |       \.
///transdata1    \.
///    |          \.
///  reshape1    transdata2
///    |     \    /
///   var1   const1
ut::GraphBuilder Graph3Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto reshape1 = builder.AddNode("reshape1", "Reshape", 2, 1);
  auto transdata2 = builder.AddNode("transdata2", "Transdata", 1, 1);
  auto transdata1 = builder.AddNode("transdata1", "Transdata", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 2, 0);

  builder.AddDataEdge(var1, 0, reshape1, 0);
  builder.AddDataEdge(const1, 0, reshape1, 1);
  builder.AddDataEdge(const1, 0, transdata2, 0);
  builder.AddDataEdge(reshape1, 0, transdata1, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 1);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);

  return builder;
}

}  // namespace

/*
TEST_F(UtestReshapeRemovePass, reshape_remove_with_const) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new ReshapeRemovePass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }

  EXPECT_EQ(graph->FindNode("reshape1"), nullptr);
  auto const1 = graph->FindNode("const1");
  EXPECT_TRUE(const1->GetOutNodes().empty());
  EXPECT_TRUE(const1->GetInNodes().empty());
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(var1->GetOutDataNodes().at(0)->GetName(), "transdata1");
}


TEST_F(UtestReshapeRemovePass, reshape_remove_without_const_two_reshape) {
  auto builder = Graph2Builder();
  auto graph = builder.GetGraph();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new ReshapeRemovePass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }

  EXPECT_EQ(graph->FindNode("reshape1"), nullptr);
  auto const1 = graph->FindNode("const1");
  EXPECT_TRUE(const1->GetOutNodes().empty());
  EXPECT_TRUE(const1->GetInNodes().empty());
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(var1->GetOutDataNodes().at(0)->GetName(), "transdata1");
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetInNodes().size(), 2);
  std::set<std::string> names;
  for (auto node : netoutput1->GetInNodes()) {
    names.insert(node->GetName());
  }
  EXPECT_EQ(names, std::set<std::string>({"var2", "transdata1"}));
}

TEST_F(UtestReshapeRemovePass, reshape_remove_without_const) {
  auto builder = Graph3Builder();
  auto graph = builder.GetGraph();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new ReshapeRemovePass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }

  EXPECT_EQ(graph->FindNode("reshape1"), nullptr);
  auto const1 = graph->FindNode("const1");
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(var1->GetOutDataNodes().at(0)->GetName(), "transdata1");
  EXPECT_NE(const1, nullptr);
  EXPECT_EQ(const1->GetOutNodes().size(), 1);
}
*/
}  // namespace ge
