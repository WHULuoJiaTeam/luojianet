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

#include <string>
#include <gtest/gtest.h>

#define private public
#include "graph/passes/enter_pass.h"
#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#include "utils/graph_utils.h"
#undef private

namespace ge {
namespace {

class UtestGraphPassesEnterPass : public testing::Test {
 protected:
  void BuildGraph() {
    // Tensor
    GeTensorDesc bool_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc scalar_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

    // const
    auto const_op_desc = std::make_shared<OpDesc>("a", "Constant");
    const_op_desc->AddOutputDesc(scalar_tensor_desc);
    auto const_node_ = graph_->AddNode(const_op_desc);

    // enter
    auto enter_op_desc = std::make_shared<OpDesc>("Enter", "Enter");
    enter_op_desc->AddInputDesc(scalar_tensor_desc);
    enter_op_desc->AddOutputDesc(scalar_tensor_desc);
    enter_node_ = graph_->AddNode(enter_op_desc);
    (void)GraphUtils::AddEdge(const_node_->GetOutDataAnchor(0), enter_node_->GetInDataAnchor(0));

    // less
    auto x_op_desc = std::make_shared<OpDesc>("x", VARIABLEV2);
    x_op_desc->AddOutputDesc(scalar_tensor_desc);
    auto x_node = graph_->AddNode(x_op_desc);
    auto y_op_desc = std::make_shared<OpDesc>("y", VARIABLEV2);
    y_op_desc->AddOutputDesc(scalar_tensor_desc);
    auto y_node = graph_->AddNode(y_op_desc);

    auto less_op_desc = std::make_shared<OpDesc>("Less", "Less");
    less_op_desc->AddInputDesc(scalar_tensor_desc);
    less_op_desc->AddInputDesc(scalar_tensor_desc);
    less_op_desc->AddOutputDesc(bool_tensor_desc);
    auto less_node = graph_->AddNode(less_op_desc);
    (void)GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(y_node->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(enter_node_->GetOutControlAnchor(), less_node->GetInControlAnchor());
  }

  ComputeGraphPtr graph_;
  EnterPass pass_;
  NodePtr enter_node_;
};
}  // namespace

TEST_F(UtestGraphPassesEnterPass, null_input) {
  NodePtr node = nullptr;
  EXPECT_EQ(pass_.Run(node), PARAM_INVALID);
}

TEST_F(UtestGraphPassesEnterPass, run_success) {
  graph_ = std::make_shared<ComputeGraph>("UTEST_graph_passes_enter_pass_run_success");
  BuildGraph();
  EXPECT_NE(enter_node_, nullptr);

  EXPECT_EQ(pass_.Run(enter_node_), SUCCESS);
  EXPECT_EQ(enter_node_->GetOutControlAnchor()->GetPeerAnchors().empty(), true);
}
}  // namespace ge
