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

#include "graph/passes/variable_prepare_op_pass.h"

#include <gtest/gtest.h>
#include <string>

using namespace ge;

class UtestGraphPassesVariablePreparePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

/// variable -- const
///   \        /
///   \       /
///     assign
TEST_F(UtestGraphPassesVariablePreparePass, variable_prepare_pass_succ1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr const_node = NodeBuilder("const", CONSTANT)
                               .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .Build(graph);

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));

  ge::VariablePrepareOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(apply_assign_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}

/// variable -- applyMoment
TEST_F(UtestGraphPassesVariablePreparePass, variable_prepare_pass_succ2) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr apply_monetum_node = NodeBuilder("apply_monetum", APPLYMOMENTUM)
                                       .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                       .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                       .Build(graph);

  ge::NodePtr sinh_node = NodeBuilder("sinh", SINH)
                              .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                              .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                              .Build(graph);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_monetum_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(apply_monetum_node->GetOutControlAnchor(), sinh_node->GetInControlAnchor());

  ge::VariablePrepareOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(apply_monetum_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}

/// variable -- const1
///   \        /
///   \       /
///   assign_add1 -- const2
///       \         /
///       \        /
///       assign_sub -- const3
///          \           /
///          \          /
///          assign_add2 -- const4
///              \           /
///              \          /
///               assign_add3
TEST_F(UtestGraphPassesVariablePreparePass, variable_prepare_pass_succ3) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);
  ge::NodePtr const_node1 = NodeBuilder("const1", CONSTANT)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr const_node2 = NodeBuilder("const2", CONSTANT)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr const_node3 = NodeBuilder("const3", CONSTANT)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr const_node4 = NodeBuilder("const4", CONSTANT)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr assign_add1 = NodeBuilder("assign_add1", ASSIGNADD)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr assign_sub = NodeBuilder("assign_sub", ASSIGNSUB)
                               .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .Build(graph);
  ge::NodePtr assign_add2 = NodeBuilder("assign_add2", ASSIGNADD)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr assign_add3 = NodeBuilder("assign_add3", ASSIGNADD)
                                .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), assign_add1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), assign_add1->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(assign_add1->GetOutDataAnchor(0), assign_sub->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), assign_sub->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(assign_sub->GetOutDataAnchor(0), assign_add2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), assign_add2->GetInDataAnchor(1));

  ge::GraphUtils::AddEdge(assign_add2->GetOutDataAnchor(0), assign_add3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node4->GetOutDataAnchor(0), assign_add3->GetInDataAnchor(1));

  ge::VariablePrepareOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(assign_add3->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}
