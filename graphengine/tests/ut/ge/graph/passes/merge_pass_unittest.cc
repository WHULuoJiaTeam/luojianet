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
#include <cstdint>
#include <memory>
#include <string>

#define private public
#include "graph/passes/merge_pass.h"

#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#undef private

namespace ge {
namespace {

class UtestGraphPassesMergePass : public testing::Test {
 protected:
  UtestGraphPassesMergePass() {
    graph_ = std::make_shared<ComputeGraph>("test");
    vector<int64_t> shape_vec{1, 1, 224, 224};
    GeShape shape = GeShape(shape_vec);
    default_tensor_desc_ = std::make_shared<GeTensorDesc>();
    default_tensor_desc_->SetShape(shape);
    default_tensor_desc_->SetFormat(FORMAT_NCHW);
    default_tensor_desc_->SetDataType(DT_FLOAT);
  }

  NodePtr NewNode(const std::string &name, const std::string &type, int input_cnt, int output_cnt) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(default_tensor_desc_->Clone());
    }

    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(default_tensor_desc_->Clone());
    }

    NodePtr node = graph_->AddNode(op_desc);
    if (type == CONSTANT) {
      int32_t weight[] = {1};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      vector<GeTensorPtr> tensor_vec = {tensor};
      OpDescUtils::SetWeights(node, tensor_vec);
    }

    return node;
  }

  ComputeGraphPtr graph_;
  GeTensorDescPtr default_tensor_desc_;
  MergePass pass_;
};

}  // namespace

TEST_F(UtestGraphPassesMergePass, null_input) {
  NodePtr node = nullptr;
  auto ret = pass_.Run(node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesMergePass, filter_non_merge_node) {
  auto node = NewNode("Op1", CONSTANT, 0, 1);
  auto ret = pass_.Run(node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesMergePass, invalid_merge_node) {
  auto merge_node = NewNode("Merge", MERGE, 2, 0);
  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

///    Op1  Op2
///     \   /
///      \ /
///     Merge
///       |
///       |
///   NetOutput
TEST_F(UtestGraphPassesMergePass, multiple_inputs) {
  auto node1 = NewNode("Op1", CONSTANT, 0, 1);
  auto node2 = NewNode("Op2", CONSTANT, 0, 1);
  auto merge_node = NewNode("Merge", MERGE, 2, 2);
  auto net_output_node = NewNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), merge_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);
}

///      Merge
///        |  \.
///        |   \.
///  Op1  Op2 Merge2
///   \    |    |
///    \   |   Op3
///     \  |   /
///     NetOutput
TEST_F(UtestGraphPassesMergePass, empty_input_cut_branch_meet_net_output_with_data_anchor) {
  auto merge_node = NewNode("Merge", MERGE, 1, 1);
  auto merge_node2 = NewNode("Merge2", MERGE, 1, 1);
  auto node1 = NewNode("Op1", CONSTANT, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), net_output->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(2));

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

///      Merge
///        |  \.
///        |   \.
///  Op1  Op2 Merge2
///   \    |    |   \.
///    \   |   Op3
///     \  |    :
///     NetOutput
TEST_F(UtestGraphPassesMergePass, empty_input_cut_branch_meet_net_output_with_control_anchor) {
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto merge_node2 = NewNode("Merge2", MERGE, 1, 2);
  auto node1 = NewNode("Op1", CONSTANT, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), net_output->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInControlAnchor());
  GraphUtils::AddEdge(merge_node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInControlAnchor());

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesMergePass, empty_input_cut_branch) {
  ///      Merge
  ///        |  \.
  ///        |   \.
  ///  Op1  Op2 Merge2
  ///   \    |    |
  ///    \   |   Op3
  ///     \  |   /
  ///      Merge3

  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto merge_node2 = NewNode("Merge2", MERGE, 1, 2);
  auto node1 = NewNode("Op1", CONSTANT, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto merge_node3 = NewNode("Merge3", MERGE, 3, 3);

  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), merge_node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge_node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), merge_node3->GetInDataAnchor(2));

  ///      Merge
  ///        |
  ///        |
  ///  Op1  Op2 Merge2
  ///   \         |
  ///    \       Op3
  ///     \      /
  ///      Merge3

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(merge_node3->GetInDataNodes().size(), 2);
  EXPECT_EQ(merge_node2->GetInDataNodes().size(), 0);
  EXPECT_EQ(node2->GetOutDataNodes().size(), 0);
  EXPECT_EQ(merge_node3->GetInDataNodes().at(0)->GetName(), "Op1");
  EXPECT_EQ(merge_node3->GetInDataNodes().at(1)->GetName(), "Op3");

  ///      Merge
  ///        |
  ///        |
  ///  Op1  Op2 Merge2
  ///   \         |
  ///    \       Op3
  ///     \.
  ///      Merge3

  ret = pass_.Run(merge_node2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(merge_node3->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node3->GetInDataNodes().at(0)->GetName(), "Op1");
  EXPECT_EQ(node3->GetOutDataNodes().size(), 0);
}

TEST_F(UtestGraphPassesMergePass, single_non_const_input) {
  ///      Op1
  ///       |
  ///     Merge
  ///      / \.
  ///    Op2 Op3
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto node1 = NewNode("Op1", RELU, 1, 1);
  auto node2 = NewNode("Op2", CONVOLUTION, 1, 1);
  auto node3 = NewNode("Op3", CONVOLUTION, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
  EXPECT_EQ(graph_->FindNode("Merge"), nullptr);
  EXPECT_EQ(node1->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node2->GetInDataNodes().size(), 1);
  EXPECT_EQ(node3->GetInDataNodes().size(), 1);
  EXPECT_EQ(node2->GetInDataAnchor(0)->GetPeerOutAnchor(), node1->GetOutDataAnchor(0));
  EXPECT_EQ(node3->GetInDataAnchor(0)->GetPeerOutAnchor(), node1->GetOutDataAnchor(0));
}

TEST_F(UtestGraphPassesMergePass, single_const_input) {
  ///     Const
  ///       |
  ///     Merge    Pass      Const
  ///      / \     ===>      /   \.
  ///    Op1 Op2           Op1   Op2
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto const_node = NewNode("Const", CONSTANT, 1, 1);
  auto node1 = NewNode("Op1", ADDN, 1, 1);
  auto node2 = NewNode("Op2", ADDN, 1, 1);

  node1->GetOpDesc()->SetIsInputConst({false});
  node2->GetOpDesc()->SetIsInputConst({false});
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
  EXPECT_EQ(graph_->FindNode("Merge").get(), nullptr);
  EXPECT_EQ(node1->GetInDataNodes().size(), 1);
  EXPECT_EQ(node2->GetInDataNodes().size(), 1);
  EXPECT_EQ(node1->GetOpDesc()->GetIsInputConst().at(0), false);
  EXPECT_EQ(node2->GetOpDesc()->GetIsInputConst().at(0), false);
}

TEST_F(UtestGraphPassesMergePass, single_const_input_value_index_two_out_nodes) {
  ///     Const
  ///       |
  ///     Merge        Pass      Const
  ///     /    |       ===>      /   \(control anchor)
  ///  Op1    | \              Op1   Constant
  ///        Op2 Op3                     |
  ///                                  /   \.
  ///                                Op2   Op3
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto const_node = NewNode("Const", CONSTANT, 1, 1);
  auto node1 = NewNode("Op1", ADDN, 1, 1);
  auto node2 = NewNode("Op2", ADDN, 1, 1);
  auto node3 = NewNode("Op3", ADDN, 1, 1);

  node1->GetOpDesc()->SetIsInputConst({false});
  node2->GetOpDesc()->SetIsInputConst({false});
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(1), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(1), node3->GetInDataAnchor(0));

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(graph_->GetDirectNodesSize(), 5);
  EXPECT_EQ(graph_->FindNode("Merge").get(), nullptr);
  EXPECT_EQ(node1->GetInDataNodes().size(), 1);
  EXPECT_EQ(node2->GetInDataNodes().size(), 1);
  EXPECT_EQ(node1->GetOpDesc()->GetIsInputConst().at(0), false);
  EXPECT_EQ(node2->GetOpDesc()->GetIsInputConst().at(0), false);

  NodePtr node_test = graph_->FindNode("Merge_value_index");
  EXPECT_NE(node_test.get(), nullptr);
  EXPECT_EQ(node_test->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_test->GetAllOutDataAnchors().size(), 1);
  EXPECT_EQ(node_test->GetInDataNodes().size(), 0);
  EXPECT_EQ(node2->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutControlAnchors().at(0),
            const_node->GetOutControlAnchor());
  EXPECT_EQ(node3->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutControlAnchors().at(0),
            const_node->GetOutControlAnchor());
  EXPECT_EQ(node2->GetInDataNodes().at(0), node_test);
  EXPECT_EQ(node3->GetInDataNodes().at(0), node_test);
}

TEST_F(UtestGraphPassesMergePass, single_const_input_value_index_two_out_nodes1) {
  ///     Const
  ///       |
  ///     Merge        Pass      Const
  ///     /    |       ===>      /   \(control anchor)
  ///  Op1    | \              Op1   Constant
  ///        Op2 Op3                     |
  ///                                  /   \.
  ///                                Op2   Op3
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto const_node = NewNode("Const", CONSTANT, 1, 1);
  auto node1 = NewNode("Op1", ADDN, 1, 1);
  auto node2 = NewNode("Op2", ADDN, 1, 1);
  auto node3 = NewNode("Op3", ADDN, 1, 1);

  node1->GetOpDesc()->SetIsInputConst({false});
  node2->GetOpDesc()->SetIsInputConst({false});
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(1), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(1), node3->GetInDataAnchor(0));

  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesMergePass, const_with_control_input) {
  ///     Switch
  ///       |
  ///    Identity
  ///       .
  ///       .
  ///       C
  ///       |
  ///     Merge
  ///      / \.
  ///    Op1 Op2
  auto switch_node = NewNode("Switch", SWITCH, 1, 2);
  auto identity_node = NewNode("Identity", SWITCH, 1, 1);
  auto const_node = NewNode("Const", CONSTANT, 1, 1);
  auto merge_node = NewNode("Merge", MERGE, 1, 2);
  auto node1 = NewNode("Op1", ADDN, 1, 1);
  auto node2 = NewNode("Op2", ADDN, 1, 1);

  node1->GetOpDesc()->SetIsInputConst({false});
  node2->GetOpDesc()->SetIsInputConst({false});
  GraphUtils::AddEdge(switch_node->GetOutDataAnchor(0), identity_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(identity_node->GetOutControlAnchor(), const_node->GetInControlAnchor());
  GraphUtils::AddEdge(identity_node->GetOutDataAnchor(0), const_node->GetInControlAnchor());
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  ///     Switch
  ///       |
  ///    Identity
  ///       .
  ///       .
  ///       C
  ///      / \.
  ///    Op1  Op2
  auto ret = pass_.Run(merge_node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 5);
  EXPECT_EQ(graph_->FindNode("Merge").get(), nullptr);
  EXPECT_EQ(node1->GetInDataNodes().size(), 1);
  EXPECT_EQ(node2->GetInDataNodes().size(), 1);
  EXPECT_EQ(node1->GetOpDesc()->GetIsInputConst().at(0), false);
  EXPECT_EQ(node2->GetOpDesc()->GetIsInputConst().at(0), false);
  EXPECT_EQ(node1->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutDataAnchors().at(0),
            identity_node->GetOutDataAnchor(0));
  EXPECT_EQ(node1->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutControlAnchors().at(0),
            identity_node->GetOutControlAnchor());
  EXPECT_EQ(node2->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutDataAnchors().at(0),
            identity_node->GetOutDataAnchor(0));
  EXPECT_EQ(node2->GetInDataNodes().at(0)->GetInControlAnchor()->GetPeerOutControlAnchors().at(0),
            identity_node->GetOutControlAnchor());
}
}  // namespace ge
