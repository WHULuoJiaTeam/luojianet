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

#include "graph/passes/fuse_data_nodes_with_common_input_pass.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <map>

#include "inc/pass_manager.h"
#include "common/ge_inner_error_codes.h"
#include "graph_builder_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {

class UtestFuseDataNodesWithCommonInputPass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}

public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, uint32_t in_num, uint32_t out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }
};

/// graph with subgraph
///       const
///       | | |
///        case
///          |
///       netoutput
///        ...
///      data0      data1       data2
TEST_F(UtestFuseDataNodesWithCommonInputPass, graph_with_subgraph1) {
  PassManager pass_manager;
  pass_manager.AddPass("FuseDataNodesWithCommonInputPass", new (std::nothrow) FuseDataNodesWithCommonInputPass);
  ComputeGraphPtr parent_graph = std::make_shared<ComputeGraph>("parent_graph");
  auto parent_const = MakeNode(parent_graph, 0, 1, "parent_const", "Const");
  auto parent_case = MakeNode(parent_graph, 3, 1, "parent_case", "Case");
  auto parent_output = MakeNode(parent_graph, 1, 0, "parent_output", "NetOutput");

  GeTensorDesc tensor_desc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);

  parent_const->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(1, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(2, tensor_desc);
  parent_case->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(1));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_case->GetInDataAnchor(2));
  GraphUtils::AddEdge(parent_case->GetOutDataAnchor(0), parent_output->GetInDataAnchor(0));

  auto case_node = parent_graph->FindNode("parent_case");
  EXPECT_NE(case_node, nullptr);
  size_t input_data_node_num = case_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 3);

  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto data0 = MakeNode(sub_graph, 1, 1, "data0", "Data");
  data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data1 = MakeNode(sub_graph, 1, 1, "data1", "Data");
  data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data2 = MakeNode(sub_graph, 1, 1, "data2", "Data");
  data2->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  (void)AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);

  sub_graph->SetParentNode(parent_case);
  sub_graph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  size_t sub_graph_num = parent_graph->GetAllSubgraphs().size();
  EXPECT_EQ(sub_graph_num, 1);

  auto data1_node = sub_graph->FindNode("data1");
  EXPECT_NE(data1_node, nullptr);
  auto data2_node = sub_graph->FindNode("data2");
  EXPECT_NE(data2_node, nullptr);

  EXPECT_EQ(pass_manager.Run(parent_graph), SUCCESS);

  // after pass, data1 and data2 are fused to data0
  data1_node = sub_graph->FindNode("data1");
  EXPECT_EQ(data1_node, nullptr);
  data2_node = sub_graph->FindNode("data2");
  EXPECT_EQ(data2_node, nullptr);
}

/// graph with subgraph
///            const
///          /       \.
///        cast1  cast1
///          \      /
///             case
///              |
///           netoutput
///        ...
///       data1       data2
///          \         /
///            add
TEST_F(UtestFuseDataNodesWithCommonInputPass, graph_with_subgraph2) {
  PassManager pass_manager;
  pass_manager.AddPass("FuseDataNodesWithCommonInputPass", new (std::nothrow) FuseDataNodesWithCommonInputPass);
  ComputeGraphPtr parent_graph = std::make_shared<ComputeGraph>("parent_graph");
  auto parent_const = MakeNode(parent_graph, 0, 1, "parent_const", "Const");
  auto parent_cast1 = MakeNode(parent_graph, 1, 1, "parent_cast1", "Cast");
  auto parent_case = MakeNode(parent_graph, 2, 1, "parent_case", "Case");
  auto parent_output = MakeNode(parent_graph, 1, 0, "parent_output", "NetOutput");

  GeTensorDesc tensor_desc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);

  parent_const->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  parent_cast1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  parent_cast1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  parent_case->GetOpDesc()->UpdateInputDesc(1, tensor_desc);
  parent_case->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_cast1->GetOutDataAnchor(0), parent_case->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_const->GetOutDataAnchor(0), parent_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(parent_cast1->GetOutDataAnchor(0), parent_case->GetInDataAnchor(1));
  GraphUtils::AddEdge(parent_case->GetOutDataAnchor(0), parent_output->GetInDataAnchor(0));

  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto data0 = MakeNode(sub_graph, 1, 1, "data0", "Data");
  data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  auto data1 = MakeNode(sub_graph, 1, 1, "data1", "Data");
  data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
  data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
  (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

  sub_graph->SetParentNode(parent_case);
  sub_graph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(sub_graph->GetName(), sub_graph);

  size_t sub_graph_num = parent_graph->GetAllSubgraphs().size();
  EXPECT_EQ(sub_graph_num, 1);
  auto data1_node = sub_graph->FindNode("data1");
  EXPECT_NE(data1_node, nullptr);

  EXPECT_EQ(pass_manager.Run(parent_graph), SUCCESS);

  // after pass, data1 is fused to data0
  data1_node = sub_graph->FindNode("data1");
  EXPECT_EQ(data1_node, nullptr);
}
}  // namespace ge
