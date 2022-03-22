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
#include <string>

#define private public

#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#include "utils/graph_utils.h"
#undef private

namespace ge {
namespace {

class UtestGraphPassesSwitchPass : public testing::Test {
 protected:
  UtestGraphPassesSwitchPass() {
    graph_ = std::make_shared<ComputeGraph>("test");
    vector<int64_t> shape_vec{1, 1, 1, 1};
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
    (void)node->SetOwnerComputeGraph(graph_);
    return node;
  }

  void BuildDefaultGraph(bool is_input_const, const bool *pred_value = nullptr) {
    ///          input  pred
    ///              \  /
    ///             Switch
    ///              |  |
    ///              F  T
    ///              |  |
    ///              Merge
    ///
    bool is_pred_const = pred_value != nullptr;
    if (is_pred_const) {
      pred_node_ = NewNode("pred", CONSTANT, 0, 1);
      int32_t weight[] = {static_cast<int32_t>(*pred_value)};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      OpDescUtils::SetWeights(pred_node_, {tensor});
    } else {
      pred_node_ = NewNode("pred", GREATER, 2, 1);
    }

    if (is_input_const) {
      int32_t weight[] = {1};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      input_node_ = NewNode("input", CONSTANT, 0, 1);
      OpDescUtils::SetWeights(input_node_, {tensor});
    } else {
      input_node_ = NewNode("input", RELU, 0, 1);
    }

    switch_node_ = NewNode("switch", SWITCH, 2, 2);
    output_false_node_ = NewNode("false_output", RELU, 1, 1);
    output_true_node_ = NewNode("true_output", RELU, 1, 1);
    merge_node_ = NewNode("merge", MERGE, 2, 1);

    switch_node_->GetOpDesc()->SetIsInputConst({false, is_pred_const});

    GraphUtils::AddEdge(input_node_->GetOutDataAnchor(0), switch_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node_->GetOutDataAnchor(0), switch_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(1), output_true_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));

    output_false_node_->GetOpDesc()->SetIsInputConst({false});
    output_true_node_->GetOpDesc()->SetIsInputConst({false});
  }

  void TestPickOutput(bool expect_output) {
    auto ret = pass_.Run(switch_node_);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(graph_->GetDirectNodesSize(), 5);  // has two isolate nodes
    EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
    if (expect_output) {
      EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor().get(), nullptr);
      EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor(), output_true_node_->GetOutDataAnchor(0));
      EXPECT_EQ(output_true_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), input_node_->GetOutDataAnchor(0));
    } else {
      EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), output_false_node_->GetOutDataAnchor(0));
      EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor().get(), nullptr);
      EXPECT_EQ(output_false_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), input_node_->GetOutDataAnchor(0));
    }
  }

  ComputeGraphPtr graph_;
  GeTensorDescPtr default_tensor_desc_;
  SwitchPass pass_;
  NodePtr pred_node_;
  NodePtr input_node_;
  NodePtr switch_node_;
  NodePtr output_false_node_;
  NodePtr output_true_node_;
  NodePtr merge_node_;
};

}  // namespace

TEST_F(UtestGraphPassesSwitchPass, null_input) {
  NodePtr node = nullptr;
  auto ret = pass_.Run(node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesSwitchPass, null_pred) {
  BuildDefaultGraph(false);
  switch_node_->GetInDataAnchor(1)->UnlinkAll();
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesSwitchPass, null_data) {
  BuildDefaultGraph(false);
  switch_node_->GetInDataAnchor(0)->UnlinkAll();
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesSwitchPass, unsupported_node_type) {
  auto node = NewNode("Op1", CONSTANT, 0, 1);
  auto ret = pass_.Run(node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesSwitchPass, empty_output) {
  BuildDefaultGraph(false);
  switch_node_->GetOutDataAnchor(0)->UnlinkAll();
  switch_node_->GetOutDataAnchor(1)->UnlinkAll();
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesSwitchPass, non_const_pred) {
  BuildDefaultGraph(false);
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_false) {
  bool pred_value = false;
  BuildDefaultGraph(false, &pred_value);
  TestPickOutput(false);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_false_float) {
  bool pred_value = false;
  BuildDefaultGraph(false, &pred_value);

  float weight[] = {0.0f};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_FLOAT);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(pred_node_, {tensor});

  TestPickOutput(false);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_false_bool) {
  bool pred_value = false;
  BuildDefaultGraph(false, &pred_value);

  bool weight[] = {false};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_BOOL);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(pred_node_, {tensor});

  TestPickOutput(false);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_false_u16) {
  bool pred_value = false;
  BuildDefaultGraph(false, &pred_value);

  uint16_t weight[] = {0};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_UINT16);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(pred_node_, {tensor});

  TestPickOutput(false);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_true) {
  bool pred_value = true;
  BuildDefaultGraph(false, &pred_value);
  TestPickOutput(true);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_true_double) {
  bool pred_value = true;
  BuildDefaultGraph(false, &pred_value);
  double weight[] = {1.0};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_DOUBLE);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(pred_node_, {tensor});

  TestPickOutput(true);
}

TEST_F(UtestGraphPassesSwitchPass, pick_output_true_int64) {
  bool pred_value = true;
  BuildDefaultGraph(false, &pred_value);
  int64_t weight[] = {1L};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT64);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(pred_node_, {tensor});

  TestPickOutput(true);
}

TEST_F(UtestGraphPassesSwitchPass, inactive_output_not_exists) {
  ///          input  pred(false)
  ///              \  /
  ///             Switch
  ///              |
  ///              F
  ///              |
  ///              Merge
  bool pred_value = false;
  BuildDefaultGraph(false, &pred_value);
  output_true_node_->GetOutDataAnchor(0)->UnlinkAll();
  GraphUtils::RemoveNodeWithoutRelink(graph_, output_true_node_);
  switch_node_->GetOutDataAnchor(1)->UnlinkAll();

  ///             input
  ///              |
  ///              F
  ///              |
  ///            Merge
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 4);
  EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), output_false_node_->GetOutDataAnchor(0));
  EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor().get(), nullptr);
  EXPECT_EQ(output_false_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), input_node_->GetOutDataAnchor(0));
}

TEST_F(UtestGraphPassesSwitchPass, const_input_pick_output_true) {
  ///            const pred(true)
  ///              \  /
  ///             Switch
  ///              |  | \
  ///              F  T1 T2
  ///              |  |  |
  ///              |  | /
  ///              |  T3
  ///              |  |
  ///              Merge
  bool pred_value = true;
  BuildDefaultGraph(true, &pred_value);
  auto output_true_node2 = NewNode("true_output2", RELU, 1, 1);
  auto output_true_node3 = NewNode("true_output3", ADD, 2, 1);
  GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(1), output_true_node2->GetInDataAnchor(0));
  GraphUtils::RemoveEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));
  GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), output_true_node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(output_true_node2->GetOutDataAnchor(0), output_true_node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(output_true_node3->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));

  ///        pred        C
  ///              |  |  |
  ///              F  T1 T2
  ///                 | /
  ///                 T3
  ///                 |
  ///               Merge
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(graph_->GetDirectNodesSize(), 7);
  EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor().get(), nullptr);
  EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor(), output_true_node3->GetOutDataAnchor(0));
  EXPECT_EQ(output_true_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), input_node_->GetOutDataAnchor(0));
  EXPECT_NE(output_true_node2->GetInDataAnchor(0)->GetPeerOutAnchor(),
            output_true_node3->GetInDataAnchor(0)->GetPeerOutAnchor());
}

TEST_F(UtestGraphPassesSwitchPass, after_switch_const_take_false_branch) {
  ///              C  pred(false)
  ///              \  /
  ///             Switch
  ///              .  .
  ///              .  .
  ///       C_1 -> F  T <- C_2
  ///              |  |
  ///              Merge
  bool pred_value = false;
  BuildDefaultGraph(true, &pred_value);
  switch_node_->GetOutDataAnchor(0)->UnlinkAll();
  switch_node_->GetOutDataAnchor(1)->UnlinkAll();

  NodePtr const_node_1 = NewNode("const_1", CONSTANT, 0, 1);
  NodePtr const_node_2 = NewNode("const_2", CONSTANT, 0, 1);
  GraphUtils::AddEdge(const_node_1->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node_2->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(0), output_false_node_->GetInControlAnchor());
  GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(1), output_true_node_->GetInControlAnchor());

  ///             C  pred(false)
  ///
  ///             C_1 C_2
  ///              |   |
  ///              F   T
  ///              |
  ///            Merge
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 7);
  EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), output_false_node_->GetOutDataAnchor(0));
  EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor().get(), nullptr);
  EXPECT_EQ(output_false_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), const_node_1->GetOutDataAnchor(0));
}

TEST_F(UtestGraphPassesSwitchPass, after_switch_const_take_true_branch) {
  ///              C  pred(true)
  ///              \  /
  ///             Switch
  ///              .  .
  ///              .  .
  ///       C_1 -> F  T <- C_2
  ///              |  |
  ///              Merge
  bool pred_value = true;
  BuildDefaultGraph(true, &pred_value);
  switch_node_->GetOutDataAnchor(0)->UnlinkAll();
  switch_node_->GetOutDataAnchor(1)->UnlinkAll();

  NodePtr const_node_1 = NewNode("const_1", CONSTANT, 0, 1);
  NodePtr const_node_2 = NewNode("const_2", CONSTANT, 0, 1);
  GraphUtils::AddEdge(const_node_1->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_node_2->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(0), output_false_node_->GetInControlAnchor());
  GraphUtils::AddEdge(switch_node_->GetOutDataAnchor(1), output_true_node_->GetInControlAnchor());

  ///              C_1 C_2
  ///              |   |
  ///              F   T
  ///                  |
  ///                Merge
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 7);
  EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor().get(), nullptr);
  EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor(), output_true_node_->GetOutDataAnchor(0));
  EXPECT_EQ(output_true_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), const_node_2->GetOutDataAnchor(0));
}

TEST_F(UtestGraphPassesSwitchPass, dead_output_connected_to_merge) {
  ///           input  pred(true)
  ///              \  /
  ///             Switch
  ///              |  |
  ///              |  T
  ///              |  |
  ///              Merge
  bool pred_value = true;
  BuildDefaultGraph(false, &pred_value);
  output_false_node_->GetOutDataAnchor(0)->UnlinkAll();
  GraphUtils::RemoveNodeWithoutRelink(graph_, output_false_node_);
  switch_node_->GetOutDataAnchor(0)->UnlinkAll();

  ///           input  pred(true)
  ///              \  /
  ///             Switch
  ///                 |
  ///                 T
  ///                 |
  ///              Merge
  auto ret = pass_.Run(switch_node_);
  EXPECT_EQ(ret, SUCCESS);

  ///                input
  ///                 |
  ///                 T
  ///                 |
  ///              Merge
  EXPECT_EQ(graph_->GetDirectNodesSize(), 4);
  EXPECT_EQ(merge_node_->GetInDataNodes().size(), 1);
  EXPECT_EQ(merge_node_->GetInDataAnchor(0)->GetPeerOutAnchor().get(), nullptr);
  EXPECT_EQ(merge_node_->GetInDataAnchor(1)->GetPeerOutAnchor(), output_true_node_->GetOutDataAnchor(0));
  EXPECT_EQ(output_true_node_->GetInDataAnchor(0)->GetPeerOutAnchor(), input_node_->GetOutDataAnchor(0));
}
}  // namespace ge
