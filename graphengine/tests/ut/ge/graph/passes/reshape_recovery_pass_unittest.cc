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

#include "graph/passes/reshape_recovery_pass.h"

#include <gtest/gtest.h>
#include <set>
#include <string>

#include "graph_builder_utils.h"

namespace ge {
class UtestReshapeRecoveryPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
///    netoutput1
///     |       \.
///transdata1    \.
///    |          \.
///    |   transdata2
///    |        /
///   var1   const1
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1, FORMAT_ND, DT_FLOAT, {-1});
  auto const1 = builder.AddNode("const1", "Const", 0, 1, FORMAT_ND, DT_FLOAT, {1, 1, 224, 224});
  auto transdata2 = builder.AddNode("transdata2", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {224, 224});
  auto transdata1 = builder.AddNode("transdata1", "Transdata", 1, 1, FORMAT_ND, DT_FLOAT, {-1, 224});
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 2, 0);

  builder.AddDataEdge(var1, 0, transdata1, 0);
  builder.AddDataEdge(const1, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 1);
  builder.AddDataEdge(transdata1, 0, netoutput1, 0);

  return builder;
}
}  // namespace

TEST_F(UtestReshapeRecoveryPass, reshape_recovery_with_dynamic_shape) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  ReshapeRecoveryPass reshape_recovery_pass;
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  Status ret = reshape_recovery_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 7);

  auto reshape1 = graph->FindNode("Reshape_ReshapeRecoveryPass_0");
  EXPECT_NE(reshape1, nullptr);
}
}  // namespace ge
