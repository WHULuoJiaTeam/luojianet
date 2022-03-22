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

#include "graph/passes/resource_pair_add_control_pass.h"

#include <set>
#include <string>
#include <gtest/gtest.h>

#include "graph_builder_utils.h"
#include "graph/passes/resource_pair_remove_control_pass.h"
#include "inc/pass_manager.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
class UtestResourcePairControlPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/// netoutput1
///    |     \.
/// StackPush  StackPop
///    |      |
///   var1   const1
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);;
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto stackpush1 = builder.AddNode("stackpush1", "StackPush", 1, 1);
  auto stackpop1 = builder.AddNode("stackpop1", "StackPop", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 2, 0);

  builder.AddDataEdge(var1, 0, stackpush1, 0);
  builder.AddDataEdge(const1, 0, stackpop1, 0);
  builder.AddDataEdge(stackpush1, 0, netoutput1, 0);
  builder.AddDataEdge(stackpop1, 0, netoutput1, 1);

  return builder;
}
}
/*
TEST_F(UtestResourcePairControlPass, resource_pair_control) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();

  auto stackpush0 = graph->FindNode("stackpush1");
  EXPECT_EQ(stackpush0->GetOutNodes().size(), 1);
  EXPECT_EQ(stackpush0->GetOutControlNodes().size(), 0);

  auto stackpop0 = graph->FindNode("stackpop1");
  EXPECT_EQ(stackpop0->GetInNodes().size(), 1);
  EXPECT_EQ(stackpop0->GetInControlNodes().size(), 0);

  ResourcePairAddControlPass add_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"", &add_pass} };
  EXPECT_EQ(PassManager::Run(graph, passes), SUCCESS);

  auto stackpush1 = graph->FindNode("stackpush1");
  EXPECT_EQ(stackpush1->GetOutNodes().size(), 2);
  EXPECT_EQ(stackpush1->GetOutControlNodes().at(0)->GetName(), "stackpop1");

  auto stackpop1 = graph->FindNode("stackpop1");
  EXPECT_EQ(stackpop1->GetInNodes().size(), 2);
  EXPECT_EQ(stackpop1->GetInControlNodes().at(0)->GetName(), "stackpush1");

  ResourcePairRemoveControlPass remove_pass;
  passes = { {"", &remove_pass} };
  EXPECT_EQ(PassManager::Run(graph, passes), SUCCESS);

  auto stackpush2 = graph->FindNode("stackpush1");
  EXPECT_EQ(stackpush2->GetOutNodes().size(), 1);
  EXPECT_EQ(stackpush2->GetOutControlNodes().size(), 0);

  auto stackpop2 = graph->FindNode("stackpop1");
  EXPECT_EQ(stackpop2->GetInNodes().size(), 1);
  EXPECT_EQ(stackpop2->GetInControlNodes().size(), 0);
}
*/
}
