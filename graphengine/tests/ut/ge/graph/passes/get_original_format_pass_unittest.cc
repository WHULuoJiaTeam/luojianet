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

#include "graph/passes/get_original_format_pass.h"

#include <gtest/gtest.h>

#include "omg/omg_inner_types.h"
#include "utils/op_desc_utils.h"

using namespace ge;
using domi::GetContext;
using domi::DOMI_TENSOR_NCHW;

class UtestGraphPassesGetOriginalFormatPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
  /// Set up a graph with the following network structure(A)
  ///             _ A _
  ///            1     2
  ///            |     |
  ///            B     C
  ///                  |
  ///                  D
  ///                  |
  ///                  E
  void make_graph(ComputeGraphPtr graph) {
    OpDescPtr op_def_a = std::make_shared<OpDesc>("A", "Data");
    OpDescPtr op_def_b = std::make_shared<OpDesc>("B", "testh");
    OpDescPtr op_def_c = std::make_shared<OpDesc>("C", "testi");
    OpDescPtr op_def_d = std::make_shared<OpDesc>("D", "Permute");
    OpDescPtr op_def_e = std::make_shared<OpDesc>("E", "testg");

    vector<int64_t> dims(4, 1);
    ge::GeShape shape(dims);
    GeTensorDesc desc_anchor(shape);

    op_def_a->AddInputDesc(desc_anchor);
    op_def_a->AddOutputDesc(desc_anchor);
    op_def_a->AddOutputDesc(desc_anchor);

    op_def_b->AddInputDesc(desc_anchor);

    op_def_c->AddInputDesc(desc_anchor);
    op_def_c->AddOutputDesc(desc_anchor);

    op_def_d->AddInputDesc(desc_anchor);

    GetContext().format = DOMI_TENSOR_NCHW;

    vector<uint32_t> permute = {0U, 2U, 3U, 1U};
    AttrUtils::SetListInt(op_def_d, PERMUTE_ATTR_ORDER, permute);

    // Add node
    NodePtr node_a = graph->AddNode(op_def_a);
    NodePtr node_b = graph->AddNode(op_def_b);
    NodePtr node_c = graph->AddNode(op_def_c);
    NodePtr node_d = graph->AddNode(op_def_d);
    NodePtr node_e = graph->AddNode(op_def_e);

    // Add edge
    GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_c->GetOutDataAnchor(1), node_d->GetInDataAnchor(0));
  }

  /// Set up a graph with the following network structure(A)
  ///              _ A _
  ///             1     2
  ///             |     |
  ///             B     C
  ///             |     |
  ///             D     E
  ///                   |
  ///                   D
  ///
  void make_invalid_graph(ComputeGraphPtr graph) {
    OpDescPtr op_def_a = std::make_shared<OpDesc>("A", "Data");
    OpDescPtr op_def_b = std::make_shared<OpDesc>("B", "testh");
    OpDescPtr op_def_c = std::make_shared<OpDesc>("C", "Permute");
    OpDescPtr op_def_d = std::make_shared<OpDesc>("D", "testd");
    OpDescPtr op_def_e = std::make_shared<OpDesc>("E", "testg");

    vector<int64_t> dims(4, 1);
    ge::GeShape shape(dims);
    GeTensorDesc desc_anchor(shape);

    op_def_a->AddInputDesc(desc_anchor);
    op_def_a->AddOutputDesc(desc_anchor);
    op_def_a->AddOutputDesc(desc_anchor);

    op_def_b->AddInputDesc(desc_anchor);
    op_def_b->AddOutputDesc(desc_anchor);

    op_def_c->AddInputDesc(desc_anchor);
    op_def_c->AddOutputDesc(desc_anchor);

    op_def_d->AddInputDesc(desc_anchor);

    GetContext().format = DOMI_TENSOR_NCHW;

    vector<uint32_t> permute = {0U, 2U, 3U, 1U};
    AttrUtils::SetListInt(op_def_d, PERMUTE_ATTR_ORDER, permute);

    // Add node
    NodePtr node_a = graph->AddNode(op_def_a);
    NodePtr node_b = graph->AddNode(op_def_b);
    NodePtr node_c = graph->AddNode(op_def_c);
    NodePtr node_d = graph->AddNode(op_def_d);
    NodePtr node_e = graph->AddNode(op_def_e);

    // Add edge
    GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_c->GetOutDataAnchor(1), node_d->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  }

  void CreateBiasaddNode(ComputeGraphPtr graph, int32_t flag) {
    // Create Biasadd Node
    OpDescPtr bias_op = std::make_shared<OpDesc>("biasadd", BIASADD);

    vector<int64_t> dim(1, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    GeTensorPtr bias = std::make_shared<ge::GeTensor>(out_desc);

    // Create convolution node
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", MATMUL);
    if (flag == 1) {
      conv_op->SetType(CONVOLUTION);
    }
    // Create mul - Node
    OpDescPtr mul_op = std::make_shared<OpDesc>("mul", MUL);

    // add descriptor
    vector<int64_t> dims(4, 1);
    GeShape shapes(dims);
    GeTensorDesc desc_anchor(shapes);

    conv_op->AddOutputDesc(desc_anchor);
    bias_op->AddInputDesc(desc_anchor);
    bias_op->AddInputDesc(desc_anchor);
    bias_op->AddOutputDesc(desc_anchor);
    mul_op->AddInputDesc(desc_anchor);

    NodePtr bias_node = graph->AddNode(bias_op);
    OpDescUtils::SetWeights(bias_node, {bias});

    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr conv_node2 = graph->AddNode(conv_op);
    NodePtr mul_node = graph->AddNode(mul_op);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node2->GetOutDataAnchor(0), bias_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), mul_node->GetInDataAnchor(0));
  }
};

TEST_F(UtestGraphPassesGetOriginalFormatPass, no_transpose_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  make_graph(graph);

  ge::GetOriginalFormatPass get_format_pass;
  Status status = get_format_pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  int32_t ori_format = 0;
  for (NodePtr n : graph->GetDirectNode()) {
    if ("Permute" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 1);
    }
    if ("testg" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 1);
    }
    if ("testh" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 0);
    }
  }
}

TEST_F(UtestGraphPassesGetOriginalFormatPass, infered_format_need_to_reset_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  make_graph(graph);
  int32_t ori_format = 1;
  for (NodePtr n : graph->GetDirectNode()) {
    if ("testh" == n->GetOpDesc()->GetType()) {
      AttrUtils::SetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
    }
    if ("Permute" == n->GetOpDesc()->GetType()) {
      vector<uint32_t> permute = {0U, 3U, 1U, 2U};
      AttrUtils::SetListInt(n->GetOpDesc(), PERMUTE_ATTR_ORDER, permute);
    }
  }

  ge::GetOriginalFormatPass get_format_pass;
  Status status = get_format_pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);

  for (NodePtr n : graph->GetDirectNode()) {
    if ("Permute" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 0);
    }
    if ("testg" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 0);
    }
    if ("testh" == n->GetOpDesc()->GetType()) {
      AttrUtils::GetInt(n->GetOpDesc(), ATTR_NAME_FORMAT, ori_format);
      EXPECT_EQ(ori_format, 1);
    }
  }
}

TEST_F(UtestGraphPassesGetOriginalFormatPass, infered_format_need_to_reset_success2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateBiasaddNode(graph, 1);

  ge::GetOriginalFormatPass get_format_pass;
  Status status = get_format_pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
}
