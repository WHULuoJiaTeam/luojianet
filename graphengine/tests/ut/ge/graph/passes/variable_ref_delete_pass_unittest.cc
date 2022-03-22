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

#include "graph/passes/variable_ref_delete_op_pass.h"

#include <gtest/gtest.h>
#include <string>

using namespace ge;

class UtestGraphPassesVariableRefDeletePass : public testing::Test {
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
///       |
///       |
///   variable_ref
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_succ1) {
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

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(apply_assign_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}

/// variable -- const
///   \        /
///   \       /
///     assign
///       |
///       |
///   variable_ref
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_fail1) {
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

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "wrong_variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(FAILED, status);
}

///     assign
///       |
///       |
///   variable_ref
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_fail2) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(FAILED, status);
}
