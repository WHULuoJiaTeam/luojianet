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

#include "graph/passes/subgraph_pass.h"
#include "inc/pass_manager.h"

namespace ge {
namespace {
class UtestGraphPassesSubgraphPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
  OpDescPtr op_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
  if (op_desc == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < input_num; i++) {
    op_desc->AddInputDesc(GeTensorDesc());
  }
  for (uint32_t i = 0; i < output_num; i++) {
    op_desc->AddOutputDesc(GeTensorDesc());
  }
  return op_desc;
}

bool CheckMemcpyExist(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      return true;
    }
  }
  return false;
}

uint32_t CheckMemcpyNum(const ComputeGraphPtr &graph) {
  uint32_t num = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == IDENTITY) {
      num++;
    }
  }
  return num;
}
} // namespace

///
/// ****** root_graph ****** * ****** subgraph branch1 ***** * ****** subgraph branch2 *****
///                          *                               *
///           Case           *              Const            *                Data
///           /   \          *                |              *                  |
///      data_0   data_1     *            NetOutput          *              NetOutput
///                          *                               *
/// ****** root_graph ****** * ****** subgraph branch1 ***** * ****** subgraph branch2 *****
///
TEST(UtestGraphPassesSubgraphPass, add_memcpy_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("add_memcpy_success");
  NodePtr func_node = graph->AddNode(CreateOpDesc("Case", CASE, 2, 1));
  NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data_0", DATA, 1, 1));
  NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data_1", DATA, 1, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), func_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), func_node->GetInDataAnchor(1)), GRAPH_SUCCESS);

  std::string subgraph_name_1 = "instance_branch_1";
  ComputeGraphPtr subgraph_1 = std::make_shared<ComputeGraph>(subgraph_name_1);
  subgraph_1->SetParentNode(func_node);
  subgraph_1->SetParentGraph(graph);
  size_t index = func_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 0);
  func_node->GetOpDesc()->AddSubgraphName("branch1");
  EXPECT_EQ(func_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);
  func_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name_1);
  EXPECT_EQ(func_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 1);

  std::string subgraph_name_2 = "instance_branch_2";
  ComputeGraphPtr subgraph_2 = std::make_shared<ComputeGraph>(subgraph_name_2);
  subgraph_2->SetParentNode(func_node);
  subgraph_2->SetParentGraph(graph);
  index = func_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  EXPECT_EQ(index, 1);
  func_node->GetOpDesc()->AddSubgraphName("branch2");
  EXPECT_EQ(func_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);
  func_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph_name_2);
  EXPECT_EQ(func_node->GetOpDesc()->GetSubgraphInstanceNames().size(), 2);

  {
    // Const->NetOutput in subgraph
    NodePtr const_node = subgraph_1->AddNode(CreateOpDesc("const", CONSTANTOP, 0, 1));
    NodePtr output_node = subgraph_1->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
  }

  {
    // Data->NetOutput in subgraph but not while body
    NodePtr data_node = subgraph_2->AddNode(CreateOpDesc("sata", DATA, 1, 1));
    NodePtr output_node = subgraph_2->AddNode(CreateOpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);
    EXPECT_TRUE(AttrUtils::SetInt(data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1));
  }

  PassManager pass_manager;
  pass_manager.AddPass("SubgraphPass", new (std::nothrow) SubgraphPass);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  EXPECT_FALSE(CheckMemcpyExist(graph));
  EXPECT_EQ(pass_manager.Run(subgraph_1), SUCCESS);
  EXPECT_EQ(CheckMemcpyNum(subgraph_1), 1);
  EXPECT_EQ(pass_manager.Run(subgraph_2), SUCCESS);
  EXPECT_EQ(CheckMemcpyNum(subgraph_2), 1);
}
}  // namespace ge
