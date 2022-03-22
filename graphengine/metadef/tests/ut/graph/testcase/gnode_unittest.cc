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

#include "graph/gnode.h"
#include "graph/node_impl.h"
#include "graph/utils/node_adapter.h"
#include "graph_builder_utils.h"
#include "graph/attr_value.h"

#define protected public
#define private public

namespace ge {
class GNodeTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(GNodeTest, GetALLSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("node", "node", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &sub_graph = sub_builder.GetGraph();
  root_graph->AddSubGraph(sub_graph);
  sub_graph->SetParentNode(node);
  sub_graph->SetParentGraph(root_graph);
  node->GetOpDesc()->AddSubgraphName("branch1");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "sub");

  std::vector<GraphPtr> subgraphs;
  ASSERT_EQ(NodeAdapter::Node2GNode(node).GetALLSubgraphs(subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs.size(), 1);
}

TEST_F(GNodeTest, GetALLSubgraphs_nullptr_root_graph) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  node->impl_->owner_graph_.reset();

  std::vector<GraphPtr> subgraphs;
  ASSERT_NE(NodeAdapter::Node2GNode(node).GetALLSubgraphs(subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(GNodeTest, GetInDataNodesAndPortIndexs_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 0, 1);
  const auto node2 = builder.AddNode("node2", "node2", 1, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
  gnode = NodeAdapter::Node2GNode(node2);
  ASSERT_NE(gnode.GetInDataNodesAndPortIndexs(0).first, nullptr);
}

TEST_F(GNodeTest, GetoutDataNodesAndPortIndexs_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 0, 1);
  const auto node2 = builder.AddNode("node2", "node2", 1, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 0);
  gnode = NodeAdapter::Node2GNode(node1);
  ASSERT_EQ(gnode.GetOutDataNodesAndPortIndexs(0).size(), 1);
}

TEST_F(GNodeTest, GetInControlNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddControlEdge(node1, node2);
  GNode gnode;
  vector<GNodePtr> in_contorl_nodes = {};
  ASSERT_EQ(gnode.GetInControlNodes().size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInControlNodes().size(), 0);
  gnode = NodeAdapter::Node2GNode(node2);
  ASSERT_EQ(gnode.GetInControlNodes().size(), 1);
}

TEST_F(GNodeTest, GetOutControlNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddControlEdge(node1, node2);
  GNode gnode;
  vector<GNodePtr> in_contorl_nodes = {};
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 0);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 0);
  gnode = NodeAdapter::Node2GNode(node1);
  ASSERT_EQ(gnode.GetOutControlNodes().size(), 1);
}

TEST_F(GNodeTest, Node2GNodePtr_success) {
  auto builder = ut::GraphBuilder("graph");
  NodePtr node = nullptr;
  ASSERT_EQ(NodeAdapter::Node2GNodePtr(node), nullptr);
  node = builder.AddNode("node", "node", 0, 0);
  ASSERT_NE(NodeAdapter::Node2GNodePtr(node), nullptr);
}

TEST_F(GNodeTest, Node2GNode2Node_success) {
  auto builder = ut::GraphBuilder("graph");
  NodePtr node = nullptr;
  ASSERT_EQ(NodeAdapter::GNode2Node(NodeAdapter::Node2GNode(node)), nullptr);
  node = builder.AddNode("node", "node", 0, 0);
  ASSERT_EQ(NodeAdapter::GNode2Node(NodeAdapter::Node2GNode(node)), node);
}

TEST_F(GNodeTest, GetName_Type_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("name", "type", 0, 0);
  GNode gnode;
  AscendString name;
  AscendString type;
  ASSERT_EQ(gnode.GetName(name), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetType(type), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetName(name), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetType(type), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  gnode.GetName(name);
  gnode.GetType(type);
  ASSERT_EQ(name, "name");
  ASSERT_EQ(type, "type");
}

TEST_F(GNodeTest, GetInputConstData_success) {
  auto sub_builder = ut::GraphBuilder("graph");
  const auto &sub_node = sub_builder.AddNode("sub_node", "Data", 3, 1);
  const auto &sub_in_data_node = sub_builder.AddNode("sub_in_data_node", "Data", 1, 1);
  const auto &sub_in_const_node = sub_builder.AddNode("sub_in_const_node", "Const", 1, 1);
  const auto &sub_in_other_node = sub_builder.AddNode("sub_in_other_node", "node_1", 0, 1);
  sub_builder.AddDataEdge(sub_in_const_node, 0, sub_node, 1);
  sub_builder.AddDataEdge(sub_in_other_node, 0, sub_node, 2);
  sub_builder.AddDataEdge(sub_in_data_node, 0, sub_node, 0);
  EXPECT_TRUE(AttrUtils::SetInt(sub_in_data_node->GetOpDesc(), "_parent_node_index", 0));
  auto root_builder = ut::GraphBuilder("graph1");
  auto root_graph = root_builder.GetGraph();
  auto sub_graph = sub_builder.GetGraph();
  const auto &root_in_node = sub_builder.AddNode("root_in_node", "Const", 0, 1); 
  const auto &root_node = sub_builder.AddNode("root_node", "Data1", 1, 1);
  root_builder.AddDataEdge(root_in_node, 0, root_node, 0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(root_node);
  GeTensor getensor;
  AttrUtils::SetTensor(root_in_node->GetOpDesc(), "value", getensor);
  AttrUtils::SetTensor(sub_in_const_node->GetOpDesc(), "value", getensor);
  root_node->GetOpDesc()->AddSubgraphName("sub_graph");
  root_node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  root_graph->AddSubgraph("sub_graph", sub_graph);
  Tensor data;
  GNode gnode;
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputConstData(0 , data), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(0 , data), GRAPH_SUCCESS);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(1 , data), GRAPH_SUCCESS);
  gnode = NodeAdapter::Node2GNode(sub_node);
  ASSERT_EQ(gnode.GetInputConstData(2 , data), GRAPH_NODE_WITHOUT_CONST_INPUT);
}

TEST_F(GNodeTest, GetInputIndexByName_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  const auto &in_other_node = builder.AddNode("in_other_node", "node_1", 0, 1);
  builder.AddDataEdge(in_other_node, 0, node, 0);
  AscendString name = nullptr;
  GNode gnode;
  int input_index;
  ASSERT_EQ(gnode.GetInputIndexByName(name , input_index), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node" , input_index), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node" , input_index), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputIndexByName("in_other_node" , input_index), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetOutputIndexByName_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 1);
  const auto &in_other_node = builder.AddNode("in_other_node", "node_1", 1, 0);
  builder.AddDataEdge(node, 0, in_other_node, 0);
  AscendString name = nullptr;
  GNode gnode;
  int input_index;
  ASSERT_EQ(gnode.GetOutputIndexByName(name , input_index), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node" , input_index), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node" , input_index), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetOutputIndexByName("in_other_node" , input_index), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetInputs_outputs_Size_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 1, 1);
  const auto &in_node = builder.AddNode("node_in", "node", 0, 1);
  const auto &out_node = builder.AddNode("node_out", "node", 1, 0);
  GNode gnode;
  ASSERT_EQ(gnode.GetInputsSize(), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputsSize(), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputsSize(), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputsSize(), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputsSize(), 1);
  ASSERT_EQ(gnode.GetOutputsSize(), 1);
}

TEST_F(GNodeTest, GetInputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  auto opdesc = node->GetOpDesc();
  GNode gnode;
  TensorDesc tensordesc;
  ASSERT_EQ(gnode.GetInputDesc(1, tensordesc), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetInputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetInputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetInputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, UpdateInputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  GNode gnode;
  const TensorDesc tensordesc;
  ASSERT_EQ(gnode.UpdateInputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.UpdateInputDesc(1, tensordesc), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.UpdateInputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.UpdateInputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetOutputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  auto opdesc = node->GetOpDesc();
  GNode gnode;
  TensorDesc tensordesc;
  ASSERT_EQ(gnode.GetOutputDesc(1, tensordesc), GRAPH_FAILED);
  ASSERT_EQ(gnode.GetOutputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetOutputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetOutputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, UpdateOutputDesc_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 1, 1);
  GNode gnode;
  const TensorDesc tensordesc;
  ASSERT_EQ(gnode.UpdateOutputDesc(-1, tensordesc), GRAPH_PARAM_INVALID);
  ASSERT_EQ(gnode.UpdateOutputDesc(1, tensordesc), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.UpdateOutputDesc(0, tensordesc), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.UpdateOutputDesc(0, tensordesc), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr1_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  vector<AscendString> attr_value;  
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  attr_value.emplace_back(name);
  name = "node";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  attr_value.clear();
  attr_value.emplace_back(name);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr2_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  attr_value = "value";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, SetAttr3_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AttrValue attr_value;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.SetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetAttr1_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = "value";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  gnode.SetAttr(name, attr_value);
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, GetAttr2_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  vector<AscendString> attr_value = {"value"};
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_PARAM_INVALID);
  name = "node";
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  gnode.SetAttr(name, attr_value);
  attr_value.clear();
  ASSERT_EQ(gnode.GetAttr(name, attr_value), GRAPH_SUCCESS);
}

TEST_F(GNodeTest, HasAttr_Success) {
  auto builder = ut::GraphBuilder("graph");
  const auto node = builder.AddNode("node", "node", 0, 0);
  GNode gnode;
  AscendString name = nullptr;
  AscendString attr_value = "value";
  ASSERT_EQ(gnode.HasAttr(name), false);
  name = "node";
  ASSERT_EQ(gnode.HasAttr(name), false);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.HasAttr(name), false);
  gnode = NodeAdapter::Node2GNode(node);
  gnode.SetAttr(name, attr_value);
  ASSERT_EQ(gnode.HasAttr(name), true);
}

TEST_F(GNodeTest, GetSubGraph_success) {
  auto sub_builder = ut::GraphBuilder("sub_graph");
  auto sub_graph = sub_builder.GetGraph();
  auto root_builder = ut::GraphBuilder("root_graph");
  const auto node = root_builder.AddNode("node", "node", 1, 1);
  auto root_graph = root_builder.GetGraph(); 
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("sub_graph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  root_graph->AddSubgraph("sub_graph", sub_graph);
  GNode gnode;
  GraphPtr graph;
  ASSERT_EQ(gnode.GetSubgraph(0U,graph), GRAPH_FAILED);
  gnode.impl_ = nullptr;
  ASSERT_EQ(gnode.GetSubgraph(0U,graph), GRAPH_FAILED);
  gnode = NodeAdapter::Node2GNode(node);
  ASSERT_EQ(gnode.GetSubgraph(0U,graph), GRAPH_SUCCESS);
}
}  // namespace ge
