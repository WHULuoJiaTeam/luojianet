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
#include <iostream>

#define protected public
#define private public
#include "graph/anchor.h"

#include "graph/node.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/graph_utils.h"
#undef protected
#undef private

using namespace ge;
using namespace std;

class UtestGeAnchor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeAnchor, data_anchor_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr_1 = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_1 = graph_ptr->AddNode(in_op_ptr_1);
  InDataAnchorPtr in_data_anchor = in_owner_node_1->GetInDataAnchor(0);

  OpDescPtr in_op_ptr_2 = std::make_shared<OpDesc>("in_op_2", "float");
  in_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddInputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddOutputDesc("z", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_2 = graph_ptr->AddNode(in_op_ptr_2);
  InDataAnchorPtr in_data_anchor_x = in_owner_node_2->GetInDataAnchor(0);
  InDataAnchorPtr in_data_anchor_y = in_owner_node_2->GetInDataAnchor(1);
  InControlAnchorPtr in_control_anchor = in_owner_node_2->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);
  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);

  EXPECT_EQ((in_data_anchor->LinkFrom(out_data_anchor_1)), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_1->LinkTo(in_data_anchor_x), GRAPH_SUCCESS);
  EXPECT_EQ(in_data_anchor_y->LinkFrom(out_data_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_control_anchor), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor->GetPeerOutDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerAnchors().size(), 2);
  EXPECT_EQ(out_data_anchor_2->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_2->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(out_data_anchor_1->GetPeerAnchors().size(), 2);
  EXPECT_NE(in_data_anchor_y->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(in_data_anchor_x->GetIdx(), 0);
  EXPECT_NE(in_data_anchor_y->GetOwnerNode(), nullptr);
  EXPECT_EQ(out_data_anchor_1->GetPeerInDataAnchors().size(), 2);
  EXPECT_EQ(in_data_anchor_x->Unlink(in_data_anchor_y), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(out_data_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->Unlink(in_data_anchor_x), GRAPH_FAILED);
  out_data_anchor_1->UnlinkAll();
  EXPECT_EQ(out_data_anchor_1->GetPeerInDataAnchors().size(), 0);
  out_data_anchor_2->UnlinkAll();
  EXPECT_EQ(out_data_anchor_2->GetPeerAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, data_anchor_exception_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node = graph_ptr->AddNode(in_op_ptr);
  InDataAnchorPtr in_data_anchor = in_owner_node->GetInDataAnchor(0);

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutDataAnchorPtr out_data_anchor_1 = out_owner_node_1->GetOutDataAnchor(0);

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutDataAnchorPtr out_data_anchor_2 = out_owner_node_2->GetOutDataAnchor(0);

  EXPECT_EQ(in_data_anchor->LinkFrom(nullptr), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InDataAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(out_data_anchor_2->LinkTo(InControlAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(nullptr), GRAPH_FAILED);
  in_data_anchor->LinkFrom(out_data_anchor_1);
  EXPECT_EQ(out_data_anchor_2->LinkTo(in_data_anchor), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->LinkFrom(out_data_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(in_data_anchor->Unlink(out_data_anchor_2), GRAPH_FAILED);
  in_data_anchor->Unlink(out_data_anchor_1);
  EXPECT_EQ(in_data_anchor->GetPeerOutAnchor(), nullptr);
}

TEST_F(UtestGeAnchor, control_anchor_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr_1 = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_1 = graph_ptr->AddNode(in_op_ptr_1);
  InControlAnchorPtr in_control_anchor_1 = in_owner_node_1->GetInControlAnchor();

  OpDescPtr in_op_ptr_2 = std::make_shared<OpDesc>("in_op_2", "float");
  in_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddInputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr_2->AddOutputDesc("z", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node_2 = graph_ptr->AddNode(in_op_ptr_2);
  InControlAnchorPtr in_control_anchor_2 = in_owner_node_2->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutControlAnchorPtr out_control_anchor_1 = out_owner_node_1->GetOutControlAnchor();

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutControlAnchorPtr out_control_anchor_2 = out_owner_node_2->GetOutControlAnchor();

  EXPECT_EQ(in_control_anchor_1->LinkFrom(out_control_anchor_1), GRAPH_SUCCESS);
  EXPECT_EQ(out_control_anchor_1->LinkTo(in_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_2->LinkFrom(out_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(in_control_anchor_2->GetPeerOutControlAnchors().size(), 2);
  EXPECT_NE(in_control_anchor_2->GetOwnerNode(), nullptr);
  EXPECT_EQ(out_control_anchor_1->GetPeerInControlAnchors().size(), 2);

  EXPECT_EQ(in_control_anchor_1->Unlink(out_control_anchor_2), GRAPH_FAILED);
  EXPECT_EQ(out_control_anchor_2->Unlink(in_control_anchor_1), GRAPH_FAILED);
  EXPECT_EQ(in_control_anchor_1->Unlink(in_control_anchor_2), GRAPH_FAILED);

  EXPECT_EQ(out_control_anchor_2->Unlink(in_control_anchor_2), GRAPH_SUCCESS);
  EXPECT_EQ(in_control_anchor_2->GetPeerOutControlAnchors().size(), 1);
  EXPECT_EQ(out_control_anchor_1->GetPeerAnchors().size(), 2);
  out_control_anchor_1->UnlinkAll();
  EXPECT_EQ(out_control_anchor_1->GetPeerAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, control_anchor_exception_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr in_op_ptr = std::make_shared<OpDesc>("in_op_1", "float");
  in_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  in_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr in_owner_node = graph_ptr->AddNode(in_op_ptr);
  InControlAnchorPtr in_control_anchor = in_owner_node->GetInControlAnchor();

  OpDescPtr out_op_ptr_1 = std::make_shared<OpDesc>("out_op_1", "float");
  out_op_ptr_1->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_1 = graph_ptr->AddNode(out_op_ptr_1);
  OutControlAnchorPtr out_control_anchor_1 = out_owner_node_1->GetOutControlAnchor();

  OpDescPtr out_op_ptr_2 = std::make_shared<OpDesc>("out_op_2", "float");
  out_op_ptr_2->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  out_op_ptr_2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr out_owner_node_2 = graph_ptr->AddNode(out_op_ptr_2);

  OutControlAnchorPtr out_control_anchor_2 = out_owner_node_2->GetOutControlAnchor();

  EXPECT_EQ(in_control_anchor->LinkFrom(nullptr), GRAPH_FAILED);
  EXPECT_EQ(out_control_anchor_1->LinkTo(InControlAnchorPtr(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(in_control_anchor->Unlink(nullptr), GRAPH_FAILED);
  in_control_anchor->LinkFrom(out_control_anchor_1);
  EXPECT_EQ(in_control_anchor->Unlink(out_control_anchor_2), GRAPH_FAILED);
  in_control_anchor->Unlink(out_control_anchor_1);
  EXPECT_EQ(in_control_anchor->GetPeerOutControlAnchors().size(), 0);
}

TEST_F(UtestGeAnchor, anchor_utils_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr relu_op_ptr = std::make_shared<OpDesc>("relu", "float");
  relu_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  relu_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr relu_node = graph_ptr->AddNode(relu_op_ptr);

  EXPECT_EQ(AnchorUtils::SetFormat(relu_node->GetInDataAnchor(0), FORMAT_NC1HWC0), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::GetFormat(relu_node->GetInDataAnchor(0)), FORMAT_NC1HWC0);

  // exception
  EXPECT_EQ(AnchorUtils::SetFormat(relu_node->GetInDataAnchor(2), FORMAT_NCHW), GRAPH_FAILED);
  EXPECT_EQ(AnchorUtils::GetFormat(relu_node->GetInDataAnchor(2)), FORMAT_RESERVED);

  EXPECT_EQ(AnchorUtils::SetFormat(relu_node->GetOutDataAnchor(0), FORMAT_NC1HWC0), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::GetFormat(relu_node->GetOutDataAnchor(0)), FORMAT_NC1HWC0);

  // exception
  EXPECT_EQ(AnchorUtils::SetFormat(relu_node->GetOutDataAnchor(0), FORMAT_RESERVED), GRAPH_FAILED);
  EXPECT_EQ(AnchorUtils::GetFormat(relu_node->GetOutDataAnchor(1)), FORMAT_RESERVED);
}

TEST_F(UtestGeAnchor, graph_utils_test) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("graph");
  OpDescPtr conv_op_ptr = std::make_shared<OpDesc>("conv", "float");
  conv_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  conv_op_ptr->AddInputDesc("w", GeTensorDesc(GeShape({32, 16, 3, 3}), FORMAT_FRACTAL_Z));
  conv_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr conv_node = graph_ptr->AddNode(conv_op_ptr);

  OpDescPtr bn_op_ptr = std::make_shared<OpDesc>("bn", "float");
  bn_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  bn_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr bn_node = graph_ptr->AddNode(bn_op_ptr);

  EXPECT_EQ(GraphUtils::AddEdge(nullptr, conv_node->GetInDataAnchor(0)), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::AddEdge(nullptr, FORMAT_NCHW, conv_node->GetInDataAnchor(0), FORMAT_NCHW), GRAPH_FAILED);

  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(
      GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), FORMAT_NC1HWC0, bn_node->GetInDataAnchor(0), FORMAT_NC1HWC0),
      GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(OutControlAnchorPtr(nullptr), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);

  OpDescPtr relu_op_ptr = std::make_shared<OpDesc>("relu", "float");
  relu_op_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  relu_op_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr relu_node = graph_ptr->AddNode(relu_op_ptr);

  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor(),
                                       relu_node->GetInControlAnchor()),
            GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor(),
                                       relu_node->GetInControlAnchor()),
            GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutControlAnchor(), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutControlAnchor(), relu_node->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0),
                                       relu_node->GetInDataAnchor(0)),
            GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::ReplaceEdgeDst(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0),
                                       relu_node->GetInDataAnchor(0)),
            GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0)), GRAPH_FAILED);

  EXPECT_EQ(GraphUtils::AddEdge(OutDataAnchorPtr(nullptr), bn_node->GetInControlAnchor()), GRAPH_FAILED);
  EXPECT_EQ(GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(GraphUtils::RemoveEdge(conv_node->GetOutDataAnchor(0), bn_node->GetInControlAnchor()), GRAPH_FAILED);
}
