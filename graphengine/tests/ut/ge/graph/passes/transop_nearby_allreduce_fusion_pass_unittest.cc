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

#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"

#include <string>
#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/passes/addn_pass.h"

namespace ge {

namespace {

class NodeBuilder {
 public:
  NodeBuilder(const string &name, const string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }
  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, Format format, DataType data_type, size_t count = 1) {
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape(vector<int64_t>(shape)));
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(data_type);
    for (int i = 0; i < count; i++) {
      op_desc_->AddInputDesc(tensor_desc);
    }
    return *this;
  }
  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, Format format, DataType data_type,
                             size_t count = 1) {
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape(vector<int64_t>(shape)));
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(data_type);
    for (int i = 0; i < count; i++) {
      op_desc_->AddOutputDesc(tensor_desc);
    }
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};

ComputeGraphPtr GetGraph1() { return nullptr; }

ComputeGraphPtr GetGraph2() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = nullptr;
  graph->AddNode(node);
  return graph;
}

ComputeGraphPtr GetGraph3() {
  // HcomAllReduce
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodeBuilder("HcomAllreduce3", HCOMALLREDUCE).Build(graph);
  return graph;
}

ComputeGraphPtr GetGraph4() {
  ///     TransData
  ///         |
  ///  HcomAllReduce
  ///         |
  ///    TransData
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph5() {
  ///       relu
  ///         |
  ///    TransData
  ///         |
  ///   HcomAllReduce
  ///        |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1 = NodeBuilder("Relu1", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  NodePtr relu2 = NodeBuilder("Relu2", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph6() {
  ///      relu
  ///        |
  ///    TransData
  ///        |
  ///  HcomAllReduce
  ///        |
  ///    TransData
  ///        |
  ///      relu
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1 = NodeBuilder("Relu1", RELU)
                      .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                      .AddOutputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                      .Build(graph);
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  NodePtr relu2 = NodeBuilder("Relu2", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph7(size_t symmetric_transdata_num, size_t asymmetric_transdata_num, size_t paired_others_num) {
  ///     TransData   TransData     ...       MatMul      ...
  ///          \         |           /       /           /
  ///                HcomAllReduce
  ///          /         |           \       \           \.
 ///     TransData   TransData     ...       RealDiv     ...
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr allreduce =
      NodeBuilder("allreduce6", HCOMALLREDUCE)
          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT, symmetric_transdata_num + asymmetric_transdata_num)
          .AddInputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT, paired_others_num)
          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT, symmetric_transdata_num + asymmetric_transdata_num)
          .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT, paired_others_num)
          .Build(graph);

  for (size_t i = 0; i < symmetric_transdata_num; i++) {
    NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                             .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);
    NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                             .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .Build(graph);
    GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(i));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i), transdata2->GetInDataAnchor(0));
  }

  for (size_t i = 0; i < asymmetric_transdata_num; i++) {
    NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                             .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                             .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);
    NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                             .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .Build(graph);
    GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(i + symmetric_transdata_num));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i + symmetric_transdata_num), transdata2->GetInDataAnchor(0));
  }

  for (size_t i = 0; i < paired_others_num; i++) {
    NodePtr matmul = NodeBuilder("matmul", MATMUL)
                         .AddInputDesc({32, 5}, FORMAT_NCHW, DT_FLOAT)
                         .AddInputDesc({32, 64}, FORMAT_NCHW, DT_FLOAT)
                         .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                         .Build(graph);
    NodePtr realDiv = NodeBuilder("realDiv", REALDIV)
                          .AddInputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
    GraphUtils::AddEdge(matmul->GetOutDataAnchor(0),
                        allreduce->GetInDataAnchor(i + symmetric_transdata_num + asymmetric_transdata_num));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i + symmetric_transdata_num + asymmetric_transdata_num),
                        realDiv->GetInDataAnchor(0));
  }
  return graph;
}

ComputeGraphPtr GetGraph8() {
  ///     TransData
  ///         |
  ///   HcomAllReduce
  ///         |
  ///     TransData
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr allreduce =
      NodeBuilder("allreduce45", HCOMALLREDUCE).AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  return graph;
}

TEST(UtestTransopNearbyAllreduceFusionPass, test1_null_graph) {
  ComputeGraphPtr graph = GetGraph1();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), INTERNAL_ERROR);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test2_null_node) {
  ComputeGraphPtr graph = GetGraph2();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test3_OnlyAllreduce) {
  ComputeGraphPtr graph = GetGraph3();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test4_all_reduce_with_trans_data) {
  ///    TransData
  ///        |
  ///  HcomAllReduce
  ///         |
  ///    TransData
  ComputeGraphPtr graph = GetGraph4();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test5_all_reduce_with_asymmetric_trans_data_and_relu) {
  ///       relu
  ///         |
  ///     TransData
  ///         |
  ///    HcomAllReduce
  ///         |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = GetGraph5();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 3);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test6_all_reduce_with_asymmetric_trans_data_and_relu) {
  ///       relu
  ///         |
  ///     TransData
  ///         |
  ///   HcomAllReduce
  ///         |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = GetGraph6();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test7_all_reduce_with_multiple_trans_datas_and_other_ops) {
  ///    TransData   TransData     ...       MatMul      ...
  ///        \         |           /       /           /
  ///              HcomAllReduce
  ///         /         |           \       \           \.
  ///    TransData   TransData     ...       RealDiv     ...
  size_t symmetric_transdata_num = 20;
  size_t asymmetric_transdata_num = 20;
  size_t paired_others_num = 20;
  ComputeGraphPtr graph = GetGraph7(symmetric_transdata_num, asymmetric_transdata_num, paired_others_num);
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), (asymmetric_transdata_num + paired_others_num) * 2 + 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test8_in_and_out_data_anchor_are_not_equal) {
  /// HcomAllReduce
  ///       |
  ///    TransData
  ComputeGraphPtr graph = GetGraph8();
  GEPass ge_pass(graph);
  graph->GetAllNodes().at(0)->SetOwnerComputeGraph(nullptr);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, FAILED);
}

}  // namespace
}  // namespace ge
