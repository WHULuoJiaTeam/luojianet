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
#include "graph/passes/addn_pass.h"

namespace ge {
namespace {

GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                            DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                             DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
    op_desc_->AddOutputDesc(tensor_desc->Clone());
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};

}  // namespace

TEST(UtestGraphPassesAddnPass, null_pass) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GEPass pass(graph);
  AddNPass *addn_pass = nullptr;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", addn_pass);
  EXPECT_EQ(pass.Run(names_to_pass), INTERNAL_ERROR);
}

TEST(UtestGraphPassesAddnPass, null_graph) {
  ComputeGraphPtr graph = nullptr;
  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), INTERNAL_ERROR);
}

TEST(UtestGraphPassesAddnPass, empty_pass) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  EXPECT_EQ(pass.Run(names_to_pass), INTERNAL_ERROR);
}

///    |
///   AddN
///    |
TEST(UtestGraphPassesAddnPass, single_addn_node) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();

  NodePtr add_n_node = NodeBuilder("add_n_node", ADDN).Build(graph);

  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &addn_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->GetDirectNodesSize(), 1);
  EXPECT_TRUE(add_n_node->GetInDataNodes().empty());
  EXPECT_TRUE(add_n_node->GetOutDataNodes().empty());
}

///   Op1
///    |
///  AddN
///    |
TEST(UtestGraphPassesAddnPass, no_output) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();

  NodePtr node = NodeBuilder("node", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr add_n_node = NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(node->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &addn_pass);
  EXPECT_NE(pass.Run(names_to_pass), SUCCESS);

  EXPECT_FALSE(add_n_node->GetInDataNodes().empty());
  EXPECT_TRUE(add_n_node->GetOutDataNodes().empty());
  EXPECT_FALSE(node->GetOutDataNodes().empty());
}

///      |
///     AddN
///      |
///     Op
TEST(UtestGraphPassesAddnPass, no_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();

  NodePtr add_n_node = NodeBuilder("add_n_node", ADDN).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr node = NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &addn_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  EXPECT_TRUE(add_n_node->GetInDataNodes().empty());
  EXPECT_EQ(node->GetInDataNodes().at(0)->GetName(), add_n_node->GetName());
}

///     Op1
///      |
///     AddN
///      |
///     Op2
TEST(UtestGraphPassesAddnPass, single_input_remove_addn_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 3);

  GEPass pass(graph);

  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &addn_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(node1->GetOutDataNodes().at(0)->GetName(), node2->GetName());
  EXPECT_EQ(node2->GetInDataNodes().at(0)->GetName(), node1->GetName());
  EXPECT_TRUE(add_n_node->GetOutDataNodes().empty());
  EXPECT_TRUE(add_n_node->GetInDataNodes().empty());
}

///     Op1  Op2
///       \  /
///       AddN
///        |
///       Op3
TEST(UtestGraphPassesAddnPass, multiple_inputs_do_not_remove) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr node2 =
      NodeBuilder("node2", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  NodePtr add_n_node = NodeBuilder("add_n_node", ADDN)
                           .AddInputDesc({1, 1, 224, 224})
                           .AddInputDesc({1, 1, 224, 224})
                           .AddOutputDesc({1, 1, 224, 224})
                           .Build(graph);

  NodePtr node3 =
      NodeBuilder("node3", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  GEPass pass(graph);
  AddNPass addn_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("Test", &addn_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
}
}  // namespace ge
