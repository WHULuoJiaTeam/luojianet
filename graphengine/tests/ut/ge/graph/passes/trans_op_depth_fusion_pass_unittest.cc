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

#include "graph/passes/transop_depth_fusion_pass.h"

#include <gtest/gtest.h>
#include <string>

using namespace ge;

class UtestGraphPassesTransOpDepthFusionPass : public testing::Test {
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

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_cast) {
  //  Node4D(fp32)->cast1(fp32->fp16)->cast2(fp16->fp32)->sinh
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);
  auto src_name = node_data->GetName();
  node_cast_1->GetOpDesc()->SetSrcName({src_name});
  node_cast_1->GetOpDesc()->SetInputName({src_name});

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);
  src_name = node_cast_1->GetName();
  node_cast_2->GetOpDesc()->SetSrcName({src_name});
  node_cast_2->GetOpDesc()->SetInputName({src_name});

  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  src_name = node_cast_2->GetName();
  node_sinh->GetOpDesc()->SetSrcName({src_name});
  node_sinh->GetOpDesc()->SetInputName({src_name});

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh->GetInDataAnchor(0));

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 2);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_cast_ctrl_edge) {
  //  Node4D(fp32)->sinh1->sinh2->cast1(fp32->fp16)->cast2(fp16->fp32)->sinh3
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // sinh1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // sinh2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh3
  ge::NodePtr node_sinh_3 = NodeBuilder("node_sinh_3", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_2->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutControlAnchor(), node_cast_1->GetInControlAnchor());
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutControlAnchor(), node_cast_2->GetInControlAnchor());

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_cast_ctrl_edge2) {
  // Node4D(fp32)->sinh1->cast1(fp32->fp16)->cast2(fp16->fp32)->sinh2->sinh3
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // sinh1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // sinh3
  ge::NodePtr node_sinh_3 = NodeBuilder("node_sinh_3", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_2->GetOutDataAnchor(0), node_sinh_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutControlAnchor(), node_sinh_3->GetInControlAnchor());

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_cast_ctrl_edge3) {
  // Node4D(fp32)->sinh1->cast1(fp32->fp16)->cast2(fp16->fp32)->sinh2->sinh3
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // sinh_1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh_2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // sinh3
  ge::NodePtr node_sinh_3 = NodeBuilder("node_sinh_3", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_2->GetOutDataAnchor(0), node_sinh_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh_3->GetInControlAnchor());

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_cast_ctrl_edge4) {
  //  Node4D(fp32)->sinh1->sinh2->cast1(fp32->fp16)->cast2(fp16->fp32)->sinh3

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // sinh1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // sinh2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh3
  ge::NodePtr node_sinh_3 = NodeBuilder("node_sinh_3", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_1->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_sinh_2->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_sinh_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_cast_2->GetInControlAnchor());

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_transpose) {
  // Node4D(NCHW)->transpose(NCHW->NHWC)->transpose(NHWC->NCHW)->sinh

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // transpose1
  ge::NodePtr node_transpose_1 = NodeBuilder("node_transpose_1", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list = {0, 2, 3, 1};
  const std::string ATTR_PERM = "perm";
  AttrUtils::SetListInt(node_transpose_1->GetOpDesc(), ATTR_PERM, order_list);

  // transpose2
  ge::NodePtr node_transpose_2 = NodeBuilder("node_transpose_2", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list2 = {0, 3, 1, 2};
  AttrUtils::SetListInt(node_transpose_2->GetOpDesc(), ATTR_PERM, order_list2);

  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_transpose_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_1->GetOutDataAnchor(0), node_transpose_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_2->GetOutDataAnchor(0), node_sinh->GetInDataAnchor(0));

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 2);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_offset_transdata) {
  // Node4D(NCHW)->transdata(NCHW->NC1HWC0)->transdata(NC1HWC0->NCHW)->sinh

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // trandata1
  ge::NodePtr node_transdata_1 = NodeBuilder("node_transdata_1", TRANSDATA)
                                     .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                                     .Build(graph);

  // transdata2
  ge::NodePtr node_transdata_2 = NodeBuilder("node_transdata_2", TRANSDATA)
                                     .AddInputDesc({2, 1, 2, 2, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                                     .AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .Build(graph);

  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transdata_1->GetOutDataAnchor(0), node_transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transdata_2->GetOutDataAnchor(0), node_sinh->GetInDataAnchor(0));

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 2);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_fold_reshape) {
  //  Node4D(NCHW)->reshape->sinh
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  // Node1D
  ge::NodePtr node_data2 = NodeBuilder("Data1D", CONSTANTOP).AddOutputDesc({4}, FORMAT_ND, DT_INT32).Build(graph);

  // reshape
  ge::NodePtr node_reshape = NodeBuilder("node_reshape", RESHAPE)
                                 .AddInputDesc({2, 16, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                 .AddInputDesc({4}, FORMAT_ND, DT_INT32)
                                 .AddOutputDesc({2, 16, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                                 .Build(graph);
  vector<AttrValue::INT> shape_v = {2, 16, 4, 1};
  AttrUtils::SetListInt(node_reshape->GetOpDesc(), RESHAPE_ATTR_SHAPE, shape_v);
  AttrUtils::SetInt(node_reshape->GetOpDesc(), RESHAPE_ATTR_AXIS, 0);
  AttrUtils::SetInt(node_reshape->GetOpDesc(), RESHAPE_ATTR_NUM_AXES, -1);

  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 16, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 16, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_reshape->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_data2->GetOutDataAnchor(0), node_reshape->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_reshape->GetOutDataAnchor(0), node_sinh->GetInDataAnchor(0));

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 3);
}

TEST_F(UtestGraphPassesTransOpDepthFusionPass, test_transop_with_multi_out_edge) {
  /// input graph
  ///
  ///                                 -->sih1
  ///                                /
  ///                     -->transpose1               -->transpose3-->sinh2
  ///                    |            \             /
  ///                    |             -->transpose2
  ///                    |                          \.
  ///                   /                            -->cast3-->cast4-->sinh3
  ///                  /
  ///                 /               -->transpose4-->transpose5-->sinh4
  ///                /               /
  ///  Node4D-->Cast1-->Cast2-->Cast5 -->reshape2-->sinh5
  ///                \               \.
  ///                 \               -->sinh6
  ///                  \.
  ///                   \            -->transpose6-->transpose7-->sinh9
  ///                    \          /
  ///                     -->reshape-->cast6-->cast7-->sinh8
  ///                              \.
  ///                                -->sinh7

  ///     after optimized graph
  ///              -->Cast4-->sinh3
  ///             /
  ///            /       -->transpose1-->sinh1
  ///           /       /
  ///          /       /-->transpose3-->sinh2
  ///          -->Cast1
  ///         /        \-->sinh7
  ///        /          \.
  ///       /            -->sinh9
  ///  Node4D
  ///       \         -->sinh4
  ///        \       /
  ///         -->Cast5-->sinh5
  ///         \      \.
  ///          \      -->sinh6
  ///           \.
  ///            -->Cast7-->sinh8
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  // Node4D
  ge::NodePtr node_data = NodeBuilder("Node4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16).Build(graph);

  // cast1
  ge::NodePtr node_cast_1 = NodeBuilder("node_cast_1", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_1->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // transpose1
  ge::NodePtr node_transpose_1 = NodeBuilder("node_transpose_1", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list1 = {0, 2, 3, 1};
  const std::string ATTR_PERM = "perm";
  AttrUtils::SetListInt(node_transpose_1->GetOpDesc(), ATTR_PERM, order_list1);

  // transpose2
  ge::NodePtr node_transpose_2 = NodeBuilder("node_transpose_2", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list2 = {0, 3, 1, 2};
  AttrUtils::SetListInt(node_transpose_2->GetOpDesc(), ATTR_PERM, order_list2);

  // sinh1
  ge::NodePtr node_sinh_1 = NodeBuilder("node_sinh_1", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  // transpose3
  ge::NodePtr node_transpose_3 = NodeBuilder("node_transpose_3", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list3 = {0, 2, 3, 1};
  AttrUtils::SetListInt(node_transpose_3->GetOpDesc(), ATTR_PERM, order_list3);

  // sinh2
  ge::NodePtr node_sinh_2 = NodeBuilder("node_sinh_2", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                .Build(graph);

  // cast3
  ge::NodePtr node_cast_3 = NodeBuilder("node_cast_3", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_3->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_3->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast4
  ge::NodePtr node_cast_4 = NodeBuilder("node_cast_4", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_4->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_4->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh3
  ge::NodePtr node_sinh_3 = NodeBuilder("node_sinh_3", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast2
  ge::NodePtr node_cast_2 = NodeBuilder("node_cast_2", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_2->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast5
  ge::NodePtr node_cast_5 = NodeBuilder("node_cast_5", CAST)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_5->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_5->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // transpose4
  ge::NodePtr node_transpose_4 = NodeBuilder("node_transpose_4", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list4 = {0, 2, 3, 1};
  AttrUtils::SetListInt(node_transpose_4->GetOpDesc(), ATTR_PERM, order_list4);

  // transpose5
  ge::NodePtr node_transpose_5 = NodeBuilder("node_transpose_5", TRANSPOSE)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list5 = {0, 3, 1, 2};
  AttrUtils::SetListInt(node_transpose_5->GetOpDesc(), ATTR_PERM, order_list5);

  // sinh4
  ge::NodePtr node_sinh_4 = NodeBuilder("node_sinh_4", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // reshape2
  ge::NodePtr node_reshape_2 = NodeBuilder("node_reshape_2", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                                   .Build(graph);
  vector<AttrValue::INT> shape_v2 = {2, 2, 4, 1};
  AttrUtils::SetListInt(node_reshape_2->GetOpDesc(), RESHAPE_ATTR_SHAPE, shape_v2);
  AttrUtils::SetInt(node_reshape_2->GetOpDesc(), RESHAPE_ATTR_AXIS, 0);
  AttrUtils::SetInt(node_reshape_2->GetOpDesc(), RESHAPE_ATTR_NUM_AXES, -1);

  // sinh5
  ge::NodePtr node_sinh_5 = NodeBuilder("node_sinh_5", SINH)
                                .AddInputDesc({2, 2, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 4, 1}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // sinh6
  ge::NodePtr node_sinh_6 = NodeBuilder("node_sinh_6", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // reshape1
  ge::NodePtr node_reshape_1 = NodeBuilder("node_reshape_1", RESHAPE)
                                   .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                   .Build(graph);
  vector<AttrValue::INT> shape_v1 = {2, 8, 1, 1};
  AttrUtils::SetListInt(node_reshape_1->GetOpDesc(), RESHAPE_ATTR_SHAPE, shape_v1);
  AttrUtils::SetInt(node_reshape_1->GetOpDesc(), RESHAPE_ATTR_AXIS, 0);
  AttrUtils::SetInt(node_reshape_1->GetOpDesc(), RESHAPE_ATTR_NUM_AXES, -1);

  // sinh7
  ge::NodePtr node_sinh_7 = NodeBuilder("node_sinh_7", SINH)
                                .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // cast6
  ge::NodePtr node_cast_6 = NodeBuilder("node_cast_6", CAST)
                                .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_6->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT);
  AttrUtils::SetInt(node_cast_6->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT16);

  // cast7
  ge::NodePtr node_cast_7 = NodeBuilder("node_cast_7", CAST)
                                .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  AttrUtils::SetInt(node_cast_7->GetOpDesc(), CAST_ATTR_SRCT, DT_FLOAT16);
  AttrUtils::SetInt(node_cast_7->GetOpDesc(), CAST_ATTR_DSTT, DT_FLOAT);

  // sinh8
  ge::NodePtr node_sinh_8 = NodeBuilder("node_sinh_8", SINH)
                                .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // transpose6
  ge::NodePtr node_transpose_6 = NodeBuilder("node_transpose_6", TRANSPOSE)
                                     .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 8, 1, 1}, FORMAT_NHWC, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list6 = {0, 2, 3, 1};
  AttrUtils::SetListInt(node_transpose_6->GetOpDesc(), ATTR_PERM, order_list6);

  // transpose7
  ge::NodePtr node_transpose_7 = NodeBuilder("node_transpose_7", TRANSPOSE)
                                     .AddInputDesc({2, 8, 1, 1}, FORMAT_NHWC, DT_FLOAT)
                                     .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                     .Build(graph);
  vector<int64_t> order_list7 = {0, 3, 1, 2};
  AttrUtils::SetListInt(node_transpose_7->GetOpDesc(), ATTR_PERM, order_list7);

  // sinh9
  ge::NodePtr node_sinh_9 = NodeBuilder("node_sinh_9", SINH)
                                .AddInputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 8, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_transpose_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_1->GetOutDataAnchor(0), node_transpose_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_1->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_2->GetOutDataAnchor(0), node_cast_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_3->GetOutDataAnchor(0), node_cast_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_4->GetOutDataAnchor(0), node_sinh_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_2->GetOutDataAnchor(0), node_transpose_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_3->GetOutDataAnchor(0), node_sinh_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_2->GetOutDataAnchor(0), node_cast_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_5->GetOutDataAnchor(0), node_transpose_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_4->GetOutDataAnchor(0), node_transpose_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_5->GetOutDataAnchor(0), node_sinh_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_5->GetOutDataAnchor(0), node_reshape_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_reshape_2->GetOutDataAnchor(0), node_sinh_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_5->GetOutDataAnchor(0), node_sinh_6->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_cast_1->GetOutDataAnchor(0), node_reshape_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_sinh_7->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_cast_6->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_6->GetOutDataAnchor(0), node_cast_7->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_cast_7->GetOutDataAnchor(0), node_sinh_8->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_transpose_6->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_6->GetOutDataAnchor(0), node_transpose_7->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_transpose_7->GetOutDataAnchor(0), node_sinh_9->GetInDataAnchor(0));

  ge::TransOpDepthFusionPass pass;
  ge::graphStatus status = pass.Run(graph);
  EXPECT_EQ(ge::GRAPH_SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 16);
  EXPECT_EQ(node_data->GetOutDataNodes().size(), 4);
  EXPECT_EQ(node_cast_1->GetOutDataNodes().size(), 4);
  EXPECT_EQ(node_cast_4->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_cast_5->GetOutDataNodes().size(), 3);
  EXPECT_EQ(node_cast_7->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_transpose_1->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_transpose_3->GetOutDataNodes().size(), 1);
}
