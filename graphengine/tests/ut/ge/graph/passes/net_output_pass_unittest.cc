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

#include "graph/passes/net_output_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphPassesNetOutputPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

ge::ComputeGraphPtr BuildClearWeightGraph(void) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr cast_op = std::make_shared<ge::OpDesc>();
  cast_op->SetType(CAST);
  cast_op->SetName("Cast1");
  cast_op->AddInputDesc(ge::GeTensorDesc());
  cast_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr cast_node = graph->AddNode(cast_op);

  ge::OpDescPtr const_op = std::make_shared<ge::OpDesc>();
  const_op->SetType(CONSTANT);
  const_op->SetName("Const1");
  const_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr const_node = graph->AddNode(const_op);

  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));

  return graph;
}

ge::ComputeGraphPtr build_graph(bool with_leaf_node = false) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  data_op->SetType(DATA);
  data_op->SetName("Data1");
  data_op->AddInputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data1 = graph->AddNode(data_op);

  ge::OpDescPtr relu_op1 = std::make_shared<ge::OpDesc>();
  relu_op1->SetType(ACTIVATION);
  relu_op1->SetName("Relu1");
  relu_op1->AddInputDesc(ge::GeTensorDesc());
  relu_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu1 = graph->AddNode(relu_op1);

  ge::OpDescPtr relu_op2 = std::make_shared<ge::OpDesc>();
  relu_op2->SetType(RELU);
  relu_op2->SetName("Relu2");
  relu_op2->AddInputDesc(ge::GeTensorDesc());
  relu_op2->AddOutputDesc(ge::GeTensorDesc());
  relu_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu2 = graph->AddNode(relu_op2);

  ge::OpDescPtr relu_op3 = std::make_shared<ge::OpDesc>();
  relu_op3->SetType(ACTIVATION);
  relu_op3->SetName("Relu3");
  relu_op3->AddInputDesc(ge::GeTensorDesc());
  relu_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu3;
  if (with_leaf_node == true) {
    relu3 = graph->AddNode(relu_op3);
  }

  ge::OpDescPtr mul_op = std::make_shared<ge::OpDesc>();
  mul_op->SetType(MUL);
  mul_op->SetName("Mul");
  mul_op->AddInputDesc(ge::GeTensorDesc());
  mul_op->AddInputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul = graph->AddNode(mul_op);

  ge::OpDescPtr mul_op1 = std::make_shared<ge::OpDesc>();
  mul_op1->SetType(MUL);
  mul_op1->SetName("Mul1");
  mul_op1->AddInputDesc(ge::GeTensorDesc());
  mul_op1->AddInputDesc(ge::GeTensorDesc());
  mul_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul1 = graph->AddNode(mul_op1);

  ge::OpDescPtr mul_op2 = std::make_shared<ge::OpDesc>();
  mul_op2->SetType(MUL);
  mul_op2->SetName("Mul2");
  mul_op2->AddInputDesc(ge::GeTensorDesc());
  mul_op2->AddInputDesc(ge::GeTensorDesc());
  mul_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul2 = graph->AddNode(mul_op2);

  ge::OpDescPtr fc_op = std::make_shared<ge::OpDesc>();
  fc_op->SetType(FULL_CONNECTION);
  fc_op->SetName("FullConnection");
  fc_op->AddInputDesc(ge::GeTensorDesc());
  fc_op->AddOutputDesc(ge::GeTensorDesc());
  fc_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr fc = graph->AddNode(fc_op);

  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), relu1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), fc->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(fc->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
  if (with_leaf_node == true) {
    ge::GraphUtils::AddEdge(fc->GetOutDataAnchor(1), relu3->GetInDataAnchor(0));
  }
  ge::GraphUtils::AddEdge(relu2->GetOutDataAnchor(0), mul->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2->GetOutDataAnchor(1), mul->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(0), mul1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(1), mul1->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(2), mul2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(3), mul2->GetInDataAnchor(1));

  return graph;
}
TEST_F(UtestGraphPassesNetOutputPass, add_ctrl_edge_for_netout_from_leaf_success) {
  ge::ComputeGraphPtr compute_graph = build_graph(true);

  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  ge::NodePtr relu3 = compute_graph->FindNode("Relu3");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{relu3, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// when output and targets set conflicts each other , output set is prio
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 1);

  std::vector<string> expect_input_data_result{"Relu3"};
  for (auto node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 2);

  std::vector<string> expect_result{"Mul1", "Mul2"};
  for (auto node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_result.begin(), expect_result.end(), name);
    if (iter != expect_result.end()) {
      expect_result.erase(iter);
    }
  }
  control_node_num = expect_result.size();
  EXPECT_EQ(control_node_num, 0);
}
TEST_F(UtestGraphPassesNetOutputPass, only_target_node_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();
  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul1, mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 0);

  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 2);

  std::vector<string> expect_result{"Mul1", "Mul2"};
  for (auto node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_result.begin(), expect_result.end(), name);
    if (iter != expect_result.end()) {
      expect_result.erase(iter);
    }
  }
  control_node_num = expect_result.size();
  EXPECT_EQ(control_node_num, 0);
}
TEST_F(UtestGraphPassesNetOutputPass, targets_with_retval_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Imitate the output node of _Retval issued
  ge::OpDescPtr retval_node_desc1 = std::make_shared<ge::OpDesc>("reval_node1", FRAMEWORKOP);
  retval_node_desc1->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc1, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc1, RETVAL_ATTR_NAME_INDEX, 0);
  ge::NodePtr retval_node1 = compute_graph->AddNode(retval_node_desc1);
  EXPECT_NE(retval_node1, nullptr);

  ge::OpDescPtr retval_node_desc2 = std::make_shared<ge::OpDesc>("reval_node2", FRAMEWORKOP);
  retval_node_desc2->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc2, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc2, RETVAL_ATTR_NAME_INDEX, 1);
  ge::NodePtr retval_node2 = compute_graph->AddNode(retval_node_desc2);
  EXPECT_NE(retval_node2, nullptr);
  // construct targets
  std::vector<ge::NodePtr> target_nodes = {retval_node1, retval_node2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);

  for (NodePtr node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node1->GetInDataAnchor(0));
    } else if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node2->GetInDataAnchor(0));
    }
  }

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 0);

  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 2);

  std::vector<string> expect_result{"Mul1", "Mul2"};
  for (auto node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_result.begin(), expect_result.end(), name);
    if (iter != expect_result.end()) {
      expect_result.erase(iter);
    }
  }
  control_node_num = expect_result.size();
  EXPECT_EQ(control_node_num, 0);
  // Check the deletion of _Retval node
  retval_node1 = compute_graph->FindNode("reval_node1");
  EXPECT_EQ(retval_node1, nullptr);
  retval_node2 = compute_graph->FindNode("reval_node2");
  EXPECT_EQ(retval_node2, nullptr);
}

TEST_F(UtestGraphPassesNetOutputPass, output_node_and_target_node_no_duplicate_success) {
  ge::ComputeGraphPtr compute_graph = build_graph(true);

  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul1, mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  ge::NodePtr relu3 = compute_graph->FindNode("Relu3");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{relu3, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// when output and targets set conflicts each other , output set is prio
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 1);

  std::vector<string> expect_input_data_result{"Relu3"};
  for (auto node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 2);

  std::vector<string> expect_result{"Mul1", "Mul2"};
  for (auto node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_result.begin(), expect_result.end(), name);
    if (iter != expect_result.end()) {
      expect_result.erase(iter);
    }
  }
  control_node_num = expect_result.size();
  EXPECT_EQ(control_node_num, 0);
}
TEST_F(UtestGraphPassesNetOutputPass, output_node_and_target_node_duplicate_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 2);

  std::vector<string> expect_input_data_result{"Mul1"};
  for (auto node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 0);
}

TEST_F(UtestGraphPassesNetOutputPass, net_output_node_and_target_node_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::OpDescPtr netout = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr netout_node = compute_graph->AddNode(netout);
  EXPECT_NE(netout_node, nullptr);

  for (NodePtr node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
    } else if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(1));
    }
  }
  // construct targets
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 1);

  std::vector<string> expect_input_data_result{"Mul1"};
  for (auto node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 1);
  std::vector<string> expect_control_data_result{"Mul2"};
  for (auto node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_control_data_result.begin(), expect_control_data_result.end(), name);
    if (iter != expect_control_data_result.end()) {
      expect_control_data_result.erase(iter);
    }
  }
  control_node_num = expect_control_data_result.size();
  EXPECT_EQ(control_node_num, 0);
}
/// graph have netoutput node.User set outputnodes and target nodes at the same time.output nodes
/// include one common node with target nodes.
/// Notice: output nodes set is more prio
TEST_F(UtestGraphPassesNetOutputPass, net_output_node_and_output_nodes_and_target_node_success_1) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::OpDescPtr netout = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr netout_node = compute_graph->AddNode(netout);
  EXPECT_NE(netout_node, nullptr);

  for (NodePtr node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
    } else if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(1));
    }
  }
  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 2);

  std::vector<string> expect_input_data_result{"Mul1", "Mul2"};
  for (auto node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 0);
}
/// graph have netoutput node.User set outputnodes and target nodes at the same time.output nodes
/// include one common node with target nodes.
/// Notice: output nodes set is more prio
TEST_F(UtestGraphPassesNetOutputPass, net_output_node_and_output_nodes_and_target_node_success_2) {
  ge::ComputeGraphPtr compute_graph = build_graph(true);

  ge::OpDescPtr netout = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr netout_node = compute_graph->AddNode(netout);
  EXPECT_NE(netout_node, nullptr);

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
    }
    if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutControlAnchor(), netout_node->GetInControlAnchor());
    }
    if (node->GetName() == "Relu3") {
      GraphUtils::AddEdge(node->GetOutControlAnchor(), netout_node->GetInControlAnchor());
    }
  }
  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 1);

  std::vector<string> expect_input_data_result{"Mul1"};
  for (const auto &node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 2);
  std::vector<string> expect_control_data_result{"Mul2", "Relu3"};
  for (const auto &node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_control_data_result.begin(), expect_control_data_result.end(), name);
    if (iter != expect_control_data_result.end()) {
      expect_control_data_result.erase(iter);
    }
  }
  control_node_num = expect_control_data_result.size();
  EXPECT_EQ(control_node_num, 0);
}
/// graph have netoutput node.User set outputnodes and target nodes at the same time.output nodes
/// include one common node with target nodes.
/// Notice: output nodes set is more prio
TEST_F(UtestGraphPassesNetOutputPass, net_output_node_and_output_nodes_and_target_node_success_3) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::OpDescPtr netout = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  netout->AddInputDesc(ge::GeTensorDesc());
  netout->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr netout_node = compute_graph->AddNode(netout);
  EXPECT_NE(netout_node, nullptr);

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
    }
    if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutControlAnchor(), netout_node->GetInControlAnchor());
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), netout_node->GetInControlAnchor());
    }
  }
  // construct targets
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<ge::NodePtr> target_nodes = {mul2};
  compute_graph->SetGraphTargetNodesInfo(target_nodes);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  // check contain netoutput
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  /// check input data node of netoutput
  /// Check data input
  int input_data_node_num = net_out_node->GetInDataNodes().size();
  EXPECT_EQ(input_data_node_num, 1);

  std::vector<string> expect_input_data_result{"Mul1"};
  for (const auto &node : net_out_node->GetInDataNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_input_data_result.begin(), expect_input_data_result.end(), name);
    if (iter != expect_input_data_result.end()) {
      expect_input_data_result.erase(iter);
    }
  }
  input_data_node_num = expect_input_data_result.size();
  EXPECT_EQ(input_data_node_num, 0);
  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 1);
  std::vector<string> expect_control_data_result{"Mul2"};
  for (const auto &node : net_out_node->GetInControlNodes()) {
    auto name = node->GetName();
    auto iter = std::find(expect_control_data_result.begin(), expect_control_data_result.end(), name);
    if (iter != expect_control_data_result.end()) {
      expect_control_data_result.erase(iter);
    }
  }
  control_node_num = expect_control_data_result.size();
  EXPECT_EQ(control_node_num, 0);
}
TEST_F(UtestGraphPassesNetOutputPass, no_output_no_target_no_retval_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Construct specified output
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphPassesNetOutputPass, no_output_no_target_no_retval_no_outnodes_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);

  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);
  EXPECT_EQ(net_out_node->GetInControlNodes().size(), 2);

  int stream_label = -1;
  EXPECT_TRUE(ge::AttrUtils::GetInt(net_out_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, stream_label));
  EXPECT_EQ(stream_label, 0);
}

TEST_F(UtestGraphPassesNetOutputPass, user_out_node_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Construct specified output
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);

  // Check data input
  string str;
  for (ge::NodePtr input_data_node : net_out_node->GetInDataNodes()) {
    str += input_data_node->GetName() + ";";
  }
  EXPECT_EQ(str, "Mul1;Mul2;");

  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();

  EXPECT_EQ(control_node_num, 0);
}

TEST_F(UtestGraphPassesNetOutputPass, retval_node_for_out_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Imitate the output node of _Retval issued
  ge::OpDescPtr retval_node_desc1 = std::make_shared<ge::OpDesc>("reval_node1", FRAMEWORKOP);
  retval_node_desc1->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc1, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc1, RETVAL_ATTR_NAME_INDEX, 0);
  ge::NodePtr retval_node1 = compute_graph->AddNode(retval_node_desc1);
  EXPECT_NE(retval_node1, nullptr);

  ge::OpDescPtr retval_node_desc2 = std::make_shared<ge::OpDesc>("reval_node2", FRAMEWORKOP);
  retval_node_desc2->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc2, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc2, RETVAL_ATTR_NAME_INDEX, 1);
  ge::NodePtr retval_node2 = compute_graph->AddNode(retval_node_desc2);
  EXPECT_NE(retval_node2, nullptr);

  for (NodePtr node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node1->GetInDataAnchor(0));
    } else if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node2->GetInDataAnchor(0));
    }
  }

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);

  // Check data input
  string str;
  for (ge::NodePtr input_data_node : net_out_node->GetInDataNodes()) {
    str += input_data_node->GetName() + ";";
  }
  EXPECT_EQ(str, "Mul1;Mul2;");

  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 0);

  // Check the deletion of _Retval node
  retval_node1 = compute_graph->FindNode("reval_node1");
  EXPECT_EQ(retval_node1, nullptr);
  retval_node2 = compute_graph->FindNode("reval_node2");
  EXPECT_EQ(retval_node2, nullptr);
}

TEST_F(UtestGraphPassesNetOutputPass, check_order_and_const_flag_success) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::OpDescPtr const_node_desc = std::make_shared<ge::OpDesc>("const_output", CONSTANT);
  const_node_desc->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr const_node = compute_graph->AddNode(const_node_desc);
  EXPECT_NE(const_node, nullptr);
  NodePtr mul1 = compute_graph->FindNode("Mul1");
  EXPECT_NE(mul1, nullptr);
  GraphUtils::AddEdge(mul1->GetOutControlAnchor(), const_node->GetInControlAnchor());

  // Construct specified output
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{const_node, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);

  ge::OpDescPtr retval_node_desc2 = std::make_shared<ge::OpDesc>("reval_node2", FRAMEWORKOP);
  retval_node_desc2->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc2, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc2, RETVAL_ATTR_NAME_INDEX, 0);
  ge::NodePtr retval_node2 = compute_graph->AddNode(retval_node_desc2);
  EXPECT_NE(retval_node2, nullptr);
  NodePtr mul2 = compute_graph->FindNode("Mul2");
  EXPECT_NE(mul2, nullptr);
  GraphUtils::AddEdge(mul2->GetOutDataAnchor(0), retval_node2->GetInDataAnchor(0));

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_NE(net_out_node, nullptr);

  // Check data input
  string str;
  for (ge::NodePtr input_data_node : net_out_node->GetInDataNodes()) {
    str += input_data_node->GetName() + ";";
  }
  EXPECT_EQ(str, "const_output;Mul2;");

  // Check control input
  int control_node_num = net_out_node->GetInControlNodes().size();
  EXPECT_EQ(control_node_num, 0);

  // Check is_input_const flag
  std::vector<bool> is_input_const = net_out_node->GetOpDesc()->GetIsInputConst();
  EXPECT_EQ(is_input_const.size(), 2);
  EXPECT_EQ(is_input_const[0], true);
  EXPECT_EQ(is_input_const[1], false);

  // Check the deletion of _Retval node
  retval_node2 = compute_graph->FindNode("reval_node2");
  EXPECT_EQ(retval_node2, nullptr);
}

/*
TEST_F(UtestGraphPassesNetOutputPass, out_node_check_fail) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Construct specified output
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_invalid_name = {{nullptr, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes_invalid_name);

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::INTERNAL_ERROR);
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_EQ(net_out_node, nullptr);

  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_invalid_index = {{mul1, 0}, {mul2, 100}};
  compute_graph->SetGraphOutNodesInfo(output_nodes_invalid_index);

  status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::INTERNAL_ERROR);
  net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_EQ(net_out_node, nullptr);
}
*/

TEST_F(UtestGraphPassesNetOutputPass, retval_node_check_fail) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Imitate the output node of _Retval issued
  ge::OpDescPtr retval_node_desc1 = std::make_shared<ge::OpDesc>("reval_node1", FRAMEWORKOP);
  retval_node_desc1->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc1, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc1, RETVAL_ATTR_NAME_INDEX, 0);
  ge::NodePtr retval_node1 = compute_graph->AddNode(retval_node_desc1);
  EXPECT_NE(retval_node1, nullptr);

  ge::OpDescPtr retval_node_desc2 = std::make_shared<ge::OpDesc>("reval_node2", FRAMEWORKOP);
  retval_node_desc2->AddInputDesc(ge::GeTensorDesc());
  (void)ge::AttrUtils::SetStr(retval_node_desc2, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_Retval");
  (void)ge::AttrUtils::SetInt(retval_node_desc2, RETVAL_ATTR_NAME_INDEX, 0);
  ge::NodePtr retval_node2 = compute_graph->AddNode(retval_node_desc2);
  EXPECT_NE(retval_node2, nullptr);

  for (NodePtr node : compute_graph->GetDirectNode()) {
    if (node->GetName() == "Mul1") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node1->GetInDataAnchor(0));
    } else if (node->GetName() == "Mul2") {
      GraphUtils::AddEdge(node->GetOutDataAnchor(0), retval_node2->GetInDataAnchor(0));
    }
  }

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::INTERNAL_ERROR);
  NodePtr net_out_node = compute_graph->FindNode(NODE_NAME_NET_OUTPUT);
  EXPECT_EQ(net_out_node, nullptr);
}

TEST_F(UtestGraphPassesNetOutputPass, out_node_update_desc_check_fail) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  ge::OpDescPtr netout = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  ge::NodePtr netout_node = compute_graph->AddNode(netout);
  EXPECT_NE(netout_node, nullptr);

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::INTERNAL_ERROR);
}

TEST_F(UtestGraphPassesNetOutputPass, out_node_remove_check_fail) {
  ge::ComputeGraphPtr compute_graph = build_graph();

  // Construct specified output
  ge::NodePtr mul1 = compute_graph->FindNode("Mul1");
  ge::NodePtr mul2 = compute_graph->FindNode("Mul2");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{mul1, 0}, {mul2, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  mul1->GetInDataAnchor(0)->UnlinkAll();
  mul1->GetInDataAnchor(1)->UnlinkAll();
  GraphUtils::RemoveNodeWithoutRelink(compute_graph, mul1);
  mul1 = compute_graph->FindNode("Mul1");
  EXPECT_EQ(mul1, nullptr);

  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) NetOutputPass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphPassesNetOutputPass, clear_weight) {
  ge::ComputeGraphPtr compute_graph = BuildClearWeightGraph();
  auto cast = compute_graph->FindNode("Cast1");
  Status ret = ge::OpDescUtils::ClearWeights(cast);
  EXPECT_EQ(ge::SUCCESS, ret);
}
