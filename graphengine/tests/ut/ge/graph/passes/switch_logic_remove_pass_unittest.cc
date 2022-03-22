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

#include "graph/passes/switch_logic_remove_pass.h"

#include <gtest/gtest.h>

#include "graph/passes/base_pass.h"
#include "graph/passes/merge_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestSwitchLogicRemovePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
///    netoutput1
///      |
///     add1
///    /   \T
/// var3    swtich2
///       T/   |
///   switch1  |
///  /       \ |
/// var1     var2
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "RefSwitch", 2, 2);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto switch2 = builder.AddNode("switch2", "Switch", 2, 2);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(var2, 0, switch1, 1);
  builder.AddDataEdge(var2, 0, switch2, 1);
  builder.AddDataEdge(switch1, 1, switch2, 0);
  builder.AddDataEdge(var3, 0, add1, 0);
  builder.AddDataEdge(switch2, 1, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput1, 0);
  return builder.GetGraph();
}

///       netoutput1
///           |
///         merge1
///        /     \.
///      /      add1
///    /        F|   \.
/// addn1    swtich2    var3
///   \F   T/   |
///   switch1  |
///  /       \ |
/// var1     var2
ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto switch2 = builder.AddNode("switch2", "Switch", 2, 2);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(var2, 0, switch1, 1);
  builder.AddDataEdge(var2, 0, switch2, 1);
  builder.AddDataEdge(switch1, 0, addn1, 0);
  builder.AddDataEdge(switch1, 1, switch2, 0);
  builder.AddDataEdge(addn1, 0, merge1, 0);
  builder.AddDataEdge(switch2, 0, add1, 1);
  builder.AddDataEdge(var3, 0, add1, 0);
  builder.AddDataEdge(add1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);
  return builder.GetGraph();
}

///    netoutput1
///      |
///     add1
///    /   \T
/// var3    swtich2
///       T/    \.
///   switch1    \.
///  /       \    \.
/// var1     var2  var4
ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto var4 = builder.AddNode("var4", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto switch2 = builder.AddNode("switch2", "Switch", 2, 2);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(var2, 0, switch1, 1);
  builder.AddDataEdge(var4, 0, switch2, 1);
  builder.AddDataEdge(switch1, 1, switch2, 0);
  builder.AddDataEdge(var3, 0, add1, 0);
  builder.AddDataEdge(switch2, 1, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput1, 0);
  return builder.GetGraph();
}

///       netoutput1
///          |
///        merge1
///        /    \.
///     add1     addn1
///    /   \T   F/
/// var3    swtich2
///       T/   |
///   switch1  |
///  /       \ |
/// var1     var2
ComputeGraphPtr BuildGraph5() {
  auto builder = ut::GraphBuilder("g5");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto switch2 = builder.AddNode("switch2", "Switch", 2, 2);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 2, 2);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(var2, 0, switch1, 1);
  builder.AddDataEdge(var2, 0, switch2, 1);
  builder.AddDataEdge(switch1, 1, switch2, 0);
  builder.AddDataEdge(var3, 0, add1, 0);
  builder.AddDataEdge(switch2, 1, add1, 1);
  builder.AddDataEdge(switch2, 0, addn1, 0);
  builder.AddDataEdge(add1, 0, merge1, 0);
  builder.AddDataEdge(addn1, 0, merge1, 1);
  builder.AddDataEdge(merge1, 0, netoutput1, 0);
  return builder.GetGraph();
}

}  // namespace

TEST_F(UtestSwitchLogicRemovePass, remove_same_true) {
  SwitchLogicRemovePass pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("SwitchLogicRemovePass", &pass);

  auto graph = BuildGraph1();
  GEPass ge_pass(graph);

  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  auto switch2 = graph->FindNode("switch2");
  EXPECT_EQ(switch2, nullptr);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 1);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "add1");
}

TEST_F(UtestSwitchLogicRemovePass, remove_different) {
  SwitchLogicRemovePass pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("SwitchLogicRemovePass", &pass);

  auto graph = BuildGraph2();
  GEPass ge_pass(graph);

  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->FindNode("switch2"), nullptr);
  auto add1 = graph->FindNode("add1");
  EXPECT_EQ(add1->GetOutNodes().size(), 0);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 2);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "addn1");
  auto merge1 = graph->FindNode("merge1");
  EXPECT_EQ(merge1->GetInNodes().size(), 1);
  EXPECT_EQ(merge1->GetInDataNodes().at(0)->GetName(), "addn1");
}

TEST_F(UtestSwitchLogicRemovePass, no_need_to_optimize) {
  SwitchLogicRemovePass pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("SwitchLogicRemovePass", &pass);

  auto graph = BuildGraph3();
  GEPass ge_pass(graph);

  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  auto switch2 = graph->FindNode("switch2");
  EXPECT_NE(switch2, nullptr);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 1);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "switch2");
  EXPECT_EQ(switch2->GetOutNodes().size(), 1);
  EXPECT_EQ(switch2->GetOutDataNodes().at(0)->GetName(), "add1");
}

TEST_F(UtestSwitchLogicRemovePass, both_true_and_false) {
  SwitchLogicRemovePass pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("SwitchLogicRemovePass", &pass);

  auto graph = BuildGraph5();
  GEPass ge_pass(graph);

  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  auto switch2 = graph->FindNode("switch2");
  EXPECT_EQ(switch2, nullptr);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 2);
  EXPECT_EQ(switch1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "add1");
  auto addn1 = graph->FindNode("addn1");
  EXPECT_EQ(addn1->GetInDataNodes().size(), 0);
  EXPECT_EQ(addn1->GetInNodes().size(), 2);
  EXPECT_EQ(addn1->GetOutNodes().size(), 0);
  auto merge1 = graph->FindNode("merge1");
  EXPECT_EQ(merge1->GetInNodes().size(), 1);
  EXPECT_EQ(merge1->GetInDataNodes().at(0)->GetName(), "add1");
}
}  // namespace ge
