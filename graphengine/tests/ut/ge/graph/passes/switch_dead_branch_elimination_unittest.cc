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

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/passes/switch_dead_branch_elimination.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestSwitchDeadBranchElimination : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
/*
 *   data1  const1
 *     \     /
 *      case1
 *        |
 *      relu1
 *        |
 *    netoutput
 */
ut::GraphBuilder ParentGraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

/*   
 *   data1   data2
 *     \      / 
 *      switch
 *     /      \
 *   relu1   relu2
 *     \      /
 *       merge
 *         |
 *     netoutput
 */
ut::GraphBuilder SwitchSubgraphBuilder(string graph_name, uint32_t num) {
  ut::GraphBuilder builder = ut::GraphBuilder(graph_name);

  string data1_name = "data1_" + std::to_string(num);
  auto data1 = builder.AddNode(data1_name, "Data", 0, 1);
  auto data1_desc = data1->GetOpDesc();
  EXPECT_NE(data1_desc, nullptr);
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);

  string data2_name = "data2_" + std::to_string(num);
  auto data2 = builder.AddNode(data2_name, "Data", 0, 1);
  auto data2_desc = data2->GetOpDesc();
  EXPECT_NE(data2_desc, nullptr);
  AttrUtils::SetInt(data2_desc, "_parent_node_index", 1);

  string switch_name = "switch_" + std::to_string(num);
  auto switch1 = builder.AddNode(switch_name, "Switch", 2, 2);

  string relu1_name = "relu1_" + std::to_string(num);
  auto relu1 = builder.AddNode(relu1_name, "Relu", 1, 1);

  string relu2_name = "relu2_" + std::to_string(num);
  auto relu2 = builder.AddNode(relu2_name, "Relu", 1, 1);

  string merge_name = "merge_" + std::to_string(num);
  auto merge = builder.AddNode(merge_name, "Merge", 2, 1);

  string output_name = "output_" + std::to_string(num);
  auto netoutput = builder.AddNode(output_name, NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(data2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, relu1, 0);
  builder.AddDataEdge(switch1, 1, relu2, 0);
  builder.AddDataEdge(relu1, 0, merge, 0);
  builder.AddDataEdge(relu2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, netoutput, 0);

  return builder;
}

void AddCaseSubgraph(ComputeGraphPtr &parent_graph, uint32_t branch_num) {
  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);

  for (uint32_t i = 0; i < branch_num; ++i) {
  	string name = "Branch_Graph_" + std::to_string(i);

  	auto builder_subgraph = SwitchSubgraphBuilder(name, i);
  	auto switch_subgraph = builder_subgraph.GetGraph();

  	case_node->GetOpDesc()->AddSubgraphName(switch_subgraph->GetName());
  	case_node->GetOpDesc()->SetSubgraphInstanceName(i, switch_subgraph->GetName());

  	switch_subgraph->SetParentNode(case_node);
  	switch_subgraph->SetParentGraph(parent_graph);
  	EXPECT_EQ(parent_graph->AddSubgraph(switch_subgraph->GetName(), switch_subgraph), GRAPH_SUCCESS);
  }
}
}  // namespace


TEST_F(UtestSwitchDeadBranchElimination, switch_dead_branch_elimination_across_case_success) {
  auto builder = ParentGraphBuilder();
  auto parent_graph = builder.GetGraph();

  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  SwitchDeadBranchElimination switch_pass;
  for (auto &subgraph : subgraphs) {
    auto switch_node = subgraph->FindFirstNodeMatchType("Switch");
    if (switch_node != nullptr) {
      EXPECT_EQ(switch_pass.Run(switch_node), SUCCESS);
    }
  }
  
  auto all_nodes = parent_graph->GetAllNodes();
  EXPECT_EQ(all_nodes.size(), 17);

  for (auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph->GetDirectNode().size(), 6);
    EXPECT_EQ(subgraph->FindFirstNodeMatchType("Switch"), nullptr);
    auto merge_node = subgraph->FindFirstNodeMatchType("Merge");
    EXPECT_NE(merge_node, nullptr);
    auto merge_innode = merge_node->GetInDataNodes();
    EXPECT_EQ(merge_innode.size(), 1);
  }
}
}  // namespace ge
