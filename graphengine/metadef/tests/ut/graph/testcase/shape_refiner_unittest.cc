/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define protected public
#define private public
#include "graph/compute_graph.h"
#include "graph/shape_refiner.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestShapeRefiner : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static NodePtr CreateNode(const ComputeGraphPtr &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  tensor.SetOriginFormat(FORMAT_NCHW);
  tensor.SetOriginDataType(DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph->AddNode(op_desc);
}

/*
 *                                 Data1
 *       sub_data1                   |                     sub_data2              sub_data3
 *           |               PartitionedCall2   ===>           |                      |
 *         relu1                     |                  PartitionedCall3   ===>     relu2
 *           |        <===   PartitionedCall1                  |                      |
 *      sub_output1                  |                     sub_output2           sub_output3
 *                               netoutput
 */
ComputeGraphPtr CreateGraphWithMultiSubgraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data = builder.AddNode("Data1", "Data", 1, 1);
  auto partcall1 = builder.AddNode("partcall1", "PartitionedCall", 1, 1);
  auto partcall2 = builder.AddNode("partcall2", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

  builder.AddDataEdge(data, 0, partcall2, 0);
  builder.AddDataEdge(partcall2, 0, partcall1, 0);
  builder.AddDataEdge(partcall1, 0, netoutput, 0);
  auto root_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder1 = ut::GraphBuilder("sub_graph1");
  auto sub_data1 = sub_builder1.AddNode("sub_data1", "Data", 1, 1);
  auto data1_desc = sub_data1->GetOpDesc();
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);
  auto sub_relu1 = sub_builder1.AddNode("sub_relu1", "Relu", 1, 1);
  auto sub_output1 = sub_builder1.AddNode("sub_output1", "NetOutput", 1, 0);
  sub_builder1.AddDataEdge(sub_data1, 0, sub_relu1, 0);
  sub_builder1.AddDataEdge(sub_relu1, 0, sub_output1, 0);
  auto subgraph1 = sub_builder1.GetGraph();

  ut::GraphBuilder sub_builder2 = ut::GraphBuilder("sub_graph2");
  auto sub_data2 = sub_builder2.AddNode("sub_data2", "Data", 1, 1);
  auto partcall3 = sub_builder2.AddNode("partcall3", "PartitionedCall", 1, 1);
  auto sub_output2 = sub_builder2.AddNode("sub_output2", "NetOutput", 1, 0);
  auto output2_desc = sub_output2->GetOpDesc();
  auto output2_desc_in = output2_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output2_desc_in, "_parent_node_index", 0);
  sub_builder2.AddDataEdge(sub_data2, 0, partcall3, 0);
  sub_builder2.AddDataEdge(partcall3, 0, sub_output2, 0);
  auto subgraph2 = sub_builder2.GetGraph();

  ut::GraphBuilder sub_builder3 = ut::GraphBuilder("sub_graph3");
  auto sub_data3 = sub_builder3.AddNode("sub_data3", "Data", 1, 1);
  auto sub_relu2 = sub_builder3.AddNode("sub_relu2", "Relu", 1, 1);
  auto sub_output3 = sub_builder3.AddNode("sub_output3", "NetOutput", 1, 0);
  auto output3_desc = sub_output3->GetOpDesc();
  auto output3_desc_in = output3_desc->MutableInputDesc(0);
  AttrUtils::SetInt(output3_desc_in, "_parent_node_index", 0);
  sub_builder3.AddDataEdge(sub_data3, 0, sub_relu2, 0);
  sub_builder3.AddDataEdge(sub_relu2, 0, sub_output3, 0);
  auto subgraph3 = sub_builder3.GetGraph();

  auto part_node1 = root_graph->FindNode("partcall1");
  auto part_desc1 = part_node1->GetOpDesc();
  part_desc1->AddSubgraphName("sub_graph1");
  part_desc1->SetSubgraphInstanceName(0, "sub_graph1");

  subgraph1->SetParentNode(part_node1);
  subgraph1->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph1", subgraph1);

  auto part_node2 = root_graph->FindNode("partcall2");
  auto part_desc2 = part_node2->GetOpDesc();
  part_desc2->AddSubgraphName("sub_graph2");
  part_desc2->SetSubgraphInstanceName(0, "sub_graph2");

  subgraph2->SetParentNode(part_node2);
  subgraph2->SetParentGraph(root_graph);
  root_graph->AddSubgraph("sub_graph2", subgraph2);

  auto part_node3 = subgraph2->FindNode("partcall3");
  auto part_desc3 = part_node3->GetOpDesc();
  part_desc3->AddSubgraphName("sub_graph3");
  part_desc3->SetSubgraphInstanceName(0, "sub_graph3");

  subgraph3->SetParentNode(part_node3);
  subgraph3->SetParentGraph(subgraph2);
  root_graph->AddSubgraph(subgraph3);

  return root_graph;
}

TEST_F(UtestShapeRefiner, infer_shape_and_type_for_running) {
  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto enter1 = CreateNode(graph, "enter", "Enter", 1, 1);

  auto op_enter = OpDescUtils::CreateOperatorFromNode(enter1);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(enter1, op_enter, true), GRAPH_SUCCESS);

  auto infershape_funcs_back = OperatorFactoryImpl::operator_infershape_funcs_;
  OperatorFactoryImpl::operator_infershape_funcs_.reset(new (std::nothrow) std::map<string, InferShapeFunc>());
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Merge", [](Operator &op) { return GRAPH_SUCCESS; });
  auto merge1 = CreateNode(graph, "merge1", "StreamMerge", 2, 2);
  auto op = OpDescUtils::CreateOperatorFromNode(merge1);
  merge1->GetOpDesc()->AddInferFunc(nullptr);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(merge1, op, true), GRAPH_SUCCESS);
  OperatorFactoryImpl::operator_infershape_funcs_ = infershape_funcs_back;
}

TEST_F(UtestShapeRefiner, CreateInferenceContext_cross_subgraph) {
  auto graph = CreateGraphWithMultiSubgraph();
  graph->SetGraphUnknownFlag(false);
  auto subgraph = graph->GetSubgraph("sub_graph1");
  auto relu = subgraph->FindNode("sub_relu1");

  EXPECT_EQ(ShapeRefiner::InferShapeAndType(relu, false), GRAPH_SUCCESS);
  auto in_data_node = relu->GetInDataNodes().at(0);
  int32_t out_idx = 0;
  std::map<NodePtr, int32_t> nodes_idx;
  auto ret = ShapeRefiner::GetRealInNodesAndIndex(in_data_node, out_idx, nodes_idx);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(nodes_idx.size(), 1);
  for (const auto &node_idx : nodes_idx) {
    EXPECT_EQ(node_idx.first->GetName(), "sub_relu2");
  }
}

TEST_F(UtestShapeRefiner, Infer_shape_and_type_failed) {
  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto enter1 = CreateNode(graph, "enter", "Enter", 1, 1);

  EXPECT_EQ(ShapeRefiner::InferShapeAndType(enter1, true), GRAPH_FAILED);
}

} // namespace ge