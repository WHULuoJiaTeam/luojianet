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

#include <set>
#include <string>

#include "framework/omg/omg_inner_types.h"
#include "common/local_context.h"
#include "graph/passes/subgraph_const_migration_pass.h"
#include "inc/pass_manager.h"
#include "register/op_registry.h"

namespace ge {
class UtestSubgraphConstMigrationPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    if (type == "Const") {
      uint64_t const_value = 101;
      auto weight = make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
      AttrUtils::SetTensor(op_desc, ge::ATTR_NAME_WEIGHTS, weight);
    }
    return graph->AddNode(op_desc);
  }

  void make_original_graph(const ComputeGraphPtr &graph) {
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    {
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
      AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
    }
    auto const1 = MakeNode(graph, 0, 1, "const1", "Const");
    {
      auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
      GraphUtils::AddEdge(data1->GetOutControlAnchor(), const1->GetInControlAnchor());
    }

    auto const2 = MakeNode(graph, 0, 1, "const2", "Const");
    {
      auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
      GraphUtils::AddEdge(data2->GetOutControlAnchor(), const2->GetInControlAnchor());
    }

    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
  }

  void make_multibatch_graph(const ComputeGraphPtr &graph) {
    auto index = MakeNode(graph, 1, 1, "index", "Data");
    auto data = MakeNode(graph, 1, 1, "data", "Data");
    auto data1 = MakeNode(graph, 1, 1, "data1", "Data");
    auto data2 = MakeNode(graph, 1, 1, "data2", "Data");
    AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

    auto case1 = MakeNode(graph, 4, 1, "case", "Case");
    GraphUtils::AddEdge(index->GetOutDataAnchor(0), case1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data->GetOutDataAnchor(0), case1->GetInDataAnchor(1));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), case1->GetInDataAnchor(2));
    GraphUtils::AddEdge(data2->GetOutDataAnchor(0), case1->GetInDataAnchor(3));
    auto output_node = MakeNode(graph, 1, 0, "output", "NetOutput");
    GraphUtils::AddEdge(case1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

    AttrUtils::SetInt(case1->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
    case1->GetOpDesc()->RegisterSubgraphIrName("branches", kDynamic);
    ComputeGraphPtr branch = std::make_shared<ComputeGraph>("test_branch");
    make_original_graph(branch);
    for (int i = 0; i < 2; ++i) {
      std::string name("_ascend_mbatch_batch_" + std::to_string(i));
      std::vector<NodePtr> input_nodes;
      std::vector<NodePtr> output_nodes;
      ComputeGraphPtr subgraph = GraphUtils::CloneGraph(branch, name, input_nodes, output_nodes);

      subgraph->SetName(name);
      subgraph->SetParentNode(case1);
      subgraph->SetParentGraph(graph);
      graph->AddSubgraph(subgraph->GetName(), subgraph);

      case1->GetOpDesc()->AddSubgraphName(name);
      case1->GetOpDesc()->SetSubgraphInstanceName(i, subgraph->GetName());
    }
  }
};

TEST_F(UtestSubgraphConstMigrationPass, subgraph_const_migration) {
  PassManager pass_manager;
  pass_manager.AddPass("SubgraphConstMigrationPass", new (std::nothrow) SubgraphConstMigrationPass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_multibatch_graph(graph);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}
}  // namespace ge