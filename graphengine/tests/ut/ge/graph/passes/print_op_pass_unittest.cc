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

#include "graph/passes/print_op_pass.h"

#include <gtest/gtest.h>

#include "omg/omg_inner_types.h"
#include "utils/op_desc_utils.h"

using domi::GetContext;
namespace ge {
class UtestGraphPassesPrintOpPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void make_graph(ComputeGraphPtr graph, bool match = true, int flag = 0) {
    auto data = std::make_shared<OpDesc>("Data", DATA);
    GeTensorDesc tensor_desc_data(GeShape({1, 1, 1, 1}));
    data->AddInputDesc(tensor_desc_data);
    data->AddOutputDesc(tensor_desc_data);
    auto data_node = graph->AddNode(data);

    auto data1 = std::make_shared<OpDesc>("Data", DATA);
    data1->AddInputDesc(tensor_desc_data);
    data1->AddOutputDesc(tensor_desc_data);
    auto data_node1 = graph->AddNode(data1);

    auto print_desc = std::make_shared<OpDesc>("Print", "Print");
    print_desc->AddInputDesc(tensor_desc_data);
    print_desc->AddInputDesc(tensor_desc_data);
    print_desc->AddOutputDesc(tensor_desc_data);
    auto print_node = graph->AddNode(print_desc);

    auto ret_val_desc = std::make_shared<OpDesc>("RetVal", "RetVal");
    ret_val_desc->AddInputDesc(tensor_desc_data);
    ret_val_desc->AddOutputDesc(tensor_desc_data);
    auto ret_val_node = graph->AddNode(ret_val_desc);

    auto ret = GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), print_node->GetInDataAnchor(0));
    ret = GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), print_node->GetInDataAnchor(1));
    ret = GraphUtils::AddEdge(print_node->GetOutDataAnchor(0), ret_val_node->GetInDataAnchor(0));
  }
};

TEST_F(UtestGraphPassesPrintOpPass, apply_success) {
  GetContext().out_nodes_map.clear();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);
  ge::PrintOpPass apply_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &apply_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesPrintOpPass, param_invalid) {
  ge::NodePtr node = nullptr;
  ge::PrintOpPass apply_pass;
  Status status = apply_pass.Run(node);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
}  // namespace ge
