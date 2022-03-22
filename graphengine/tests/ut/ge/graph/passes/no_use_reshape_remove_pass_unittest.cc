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

#include "graph/passes/no_use_reshape_remove_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/op_desc.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "inc/pass_manager.h"
#include "opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphNoUseReshapeRemovePass : public testing::Test {
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

/// data->expanddim->reshape1->squeeze->reshape4->sinh
///                                      /
///                                    const
void make_reshape_graph(ComputeGraphPtr &graph) {
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_expanddim_1 = NodeBuilder("ExpandDim", EXPANDDIMS)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .Build(graph);

  ge::NodePtr node_reshape_1 = NodeBuilder("Reshape_1", RESHAPE)
                                   .AddInputDesc({2, 1, 2, 2}, FORMAT_ND, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_ND, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_squeeze_1 = NodeBuilder("Squeeze", SQUEEZE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);
  ge::NodePtr node_const =
      NodeBuilder("const", CONSTANT).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_reshape_4 = NodeBuilder("Reshape_4", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_sinh_1 = NodeBuilder("sinh", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_expanddim_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_expanddim_1->GetOutDataAnchor(0), node_reshape_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_squeeze_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_squeeze_1->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_reshape_4->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
}

TEST_F(UtestGraphNoUseReshapeRemovePass, node_to_be_delete_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_reshape_graph(compute_graph);

  // normal case
  NoUseReshapeRemovePass no_use_reshape_pass;
  ge::NodePtr reshape_node = compute_graph->FindNode("Reshape_4");
  Status status = no_use_reshape_pass.Run(reshape_node);
  EXPECT_EQ(status, ge::SUCCESS);

  // not reshape node case
  ge::NodePtr squeeze_node = compute_graph->FindNode("Squeeze");
  status = no_use_reshape_pass.Run(squeeze_node);
  EXPECT_EQ(status, ge::SUCCESS);

  // ND
  ge::NodePtr reshape_node2 = compute_graph->FindNode("Reshape_1");
  status = no_use_reshape_pass.Run(reshape_node2);
  EXPECT_EQ(status, ge::SUCCESS);
}
