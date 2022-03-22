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
#include <vector>

#include "omg/omg_inner_types.h"

#define protected public
#define private public
#include "graph/passes/prune_pass.h"

#include "anchor.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace std;

class UtestGraphPassesPrunePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// case1:no net_out_put_node
TEST_F(UtestGraphPassesPrunePass, no_net_out_put_node) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  reverse_op->SetType(REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  floor_op->SetType(FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"prune_pass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  EXPECT_EQ(ge::SUCCESS, status);

  uint64_t size = graph->GetDirectNode().size();
  EXPECT_EQ(size, size_ori);
}
// case2: one net path with one bypass branch
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_only_one_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  reverse_op->SetType(REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  floor_op->SetType(FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::OpDescPtr reverse_op1 = std::make_shared<ge::OpDesc>();
  reverse_op->SetType(REVERSE);
  reverse_op->SetName("Reverse1");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node1 = graph->AddNode(reverse_op1);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"prune_pass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  uint64_t size = graph->GetDirectNode().size();
  int diff = size_ori - size;
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(diff, 1);
}
// case3: one net path with one bypass branch
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_one_valid_path_and_one_bypass_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  // valid path construct (reverse->floor->net_out)
  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  reverse_op->SetType(REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  floor_op->SetType(FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  // incvalid path construct (reverse->floor1->floor2)
  ge::OpDescPtr floor_op1 = std::make_shared<ge::OpDesc>();
  floor_op1->SetType(FLOOR);
  floor_op1->SetName("Floor1");
  floor_op1->AddInputDesc(ge::GeTensorDesc());
  floor_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node1 = graph->AddNode(floor_op1);

  ge::OpDescPtr floor_op2 = std::make_shared<ge::OpDesc>();
  floor_op2->SetType(FLOOR);
  floor_op2->SetName("Floor2");
  floor_op2->AddInputDesc(ge::GeTensorDesc());
  floor_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node2 = graph->AddNode(floor_op2);
  // isolated node
  ge::OpDescPtr floor_op3 = std::make_shared<ge::OpDesc>();
  floor_op3->SetType(FLOOR);
  floor_op3->SetName("Floor3");
  floor_op3->AddInputDesc(ge::GeTensorDesc());
  floor_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node3 = graph->AddNode(floor_op3);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(1), floor_node1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node1->GetOutDataAnchor(0), floor_node2->GetInDataAnchor(0));

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  vector<GraphPass *> passes = {&prune_pass};
}

// case 4: multi net path with one common netout(1:multi：1)
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_multi_path) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  data_op->SetType(DATA);
  data_op->SetName("data");
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op);

  ge::OpDescPtr reverse_op1 = std::make_shared<ge::OpDesc>();
  reverse_op1->SetType(REVERSE);
  reverse_op1->SetName("Reverse1");
  reverse_op1->AddInputDesc(ge::GeTensorDesc());
  reverse_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node1 = graph->AddNode(reverse_op1);

  ge::OpDescPtr floor_op1 = std::make_shared<ge::OpDesc>();
  floor_op1->SetType(FLOOR);
  floor_op1->SetName("Floor1");
  floor_op1->AddInputDesc(ge::GeTensorDesc());
  floor_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node1 = graph->AddNode(floor_op1);

  ge::OpDescPtr reverse_op2 = std::make_shared<ge::OpDesc>();
  reverse_op2->SetType(REVERSE);
  reverse_op2->SetName("Reverse2");
  reverse_op2->AddInputDesc(ge::GeTensorDesc());
  reverse_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node2 = graph->AddNode(reverse_op2);

  ge::OpDescPtr floor_op2 = std::make_shared<ge::OpDesc>();
  floor_op2->SetType(FLOOR);
  floor_op2->SetName("Floor2");
  floor_op2->AddInputDesc(ge::GeTensorDesc());
  floor_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node2 = graph->AddNode(floor_op2);

  ge::OpDescPtr reverse_op3 = std::make_shared<ge::OpDesc>();
  reverse_op3->SetType(REVERSE);
  reverse_op3->SetName("Reverse3");
  reverse_op3->AddInputDesc(ge::GeTensorDesc());
  reverse_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node3 = graph->AddNode(reverse_op3);

  ge::OpDescPtr floor_op3 = std::make_shared<ge::OpDesc>();
  floor_op3->SetType(FLOOR);
  floor_op3->SetName("Floor3");
  floor_op3->AddInputDesc(ge::GeTensorDesc());
  floor_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node3 = graph->AddNode(floor_op3);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reverse_node1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), reverse_node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(2), reverse_node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reverse_node1->GetOutDataAnchor(0), floor_node1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reverse_node2->GetOutDataAnchor(0), floor_node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node2->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(reverse_node3->GetOutDataAnchor(0), floor_node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node3->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(2));

  uint64_t size_ori = graph->GetDirectNode().size();

  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"prune_pass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  uint64_t size_after_proc = graph->GetDirectNode().size();
  EXPECT_EQ(size_ori, size_after_proc);
}
// case 5: circle,diamand style
TEST_F(UtestGraphPassesPrunePass, multi_net_out_put_node_with_circle_net) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  data_op->SetType(DATA);
  data_op->SetName("data");
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op);

  ge::OpDescPtr op_1 = std::make_shared<ge::OpDesc>();
  op_1->SetType(REVERSE);
  op_1->SetName("Reverse1");
  op_1->AddInputDesc(ge::GeTensorDesc());
  op_1->AddOutputDesc(ge::GeTensorDesc());
  op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_1 = graph->AddNode(op_1);

  ge::OpDescPtr op_2 = std::make_shared<ge::OpDesc>();
  op_2->SetType(REVERSE);
  op_2->SetName("Reverse2");
  op_2->AddInputDesc(ge::GeTensorDesc());
  op_2->AddInputDesc(ge::GeTensorDesc());
  op_2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_2 = graph->AddNode(op_2);

  ge::OpDescPtr op_3 = std::make_shared<ge::OpDesc>();
  op_3->SetType(REVERSE);
  op_3->SetName("Reverse3");
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_3 = graph->AddNode(op_3);

  ge::OpDescPtr op_4 = std::make_shared<ge::OpDesc>();
  op_4->SetType(REVERSE);
  op_4->SetName("Reverse4");
  op_4->AddInputDesc(ge::GeTensorDesc());
  op_4->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_4 = graph->AddNode(op_4);

  ge::OpDescPtr op_5 = std::make_shared<ge::OpDesc>();
  op_5->SetType(REVERSE);
  op_5->SetName("Reverse5");
  op_5->AddInputDesc(ge::GeTensorDesc());
  op_5->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_5 = graph->AddNode(op_5);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), node_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_3->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(0), node_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(1), node_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_2->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(0), node_3->GetInDataAnchor(1));

  uint64_t size_ori = graph->GetDirectNode().size();

  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"prune_pass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(ge::SUCCESS, status);
  uint64_t size_after_proc = graph->GetDirectNode().size();
  EXPECT_EQ(size_ori, size_after_proc);
}

// case 6: two mix circle and multi path,diamand style
TEST_F(UtestGraphPassesPrunePass, mix_two_circle_net) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  data_op->SetType(DATA);
  data_op->SetName("data");
  data_op->AddOutputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op);

  ge::OpDescPtr op_1 = std::make_shared<ge::OpDesc>();
  op_1->SetType(REVERSE);
  op_1->SetName("Reverse1");
  op_1->AddInputDesc(ge::GeTensorDesc());
  op_1->AddInputDesc(ge::GeTensorDesc());
  op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_1 = graph->AddNode(op_1);

  ge::OpDescPtr op_2 = std::make_shared<ge::OpDesc>();
  op_2->SetType(REVERSE);
  op_2->SetName("Reverse2");
  op_2->AddInputDesc(ge::GeTensorDesc());
  op_2->AddOutputDesc(ge::GeTensorDesc());
  op_2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_2 = graph->AddNode(op_2);

  ge::OpDescPtr op_3 = std::make_shared<ge::OpDesc>();
  op_3->SetType(REVERSE);
  op_3->SetName("Reverse3");
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddInputDesc(ge::GeTensorDesc());
  op_3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_3 = graph->AddNode(op_3);

  ge::OpDescPtr op_4 = std::make_shared<ge::OpDesc>();
  op_4->SetType(REVERSE);
  op_4->SetName("Reverse4");
  op_4->AddInputDesc(ge::GeTensorDesc());
  op_4->AddInputDesc(ge::GeTensorDesc());
  op_4->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_4 = graph->AddNode(op_4);

  ge::OpDescPtr op_5 = std::make_shared<ge::OpDesc>();
  op_5->SetType(REVERSE);
  op_5->SetName("Reverse5");
  op_5->AddInputDesc(ge::GeTensorDesc());
  op_5->AddOutputDesc(ge::GeTensorDesc());
  op_5->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_5 = graph->AddNode(op_5);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);

  ge::GraphUtils::AddEdge(node_1->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(0), node_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(0), node_1->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(0), node_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_2->GetOutDataAnchor(1), node_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5->GetOutDataAnchor(1), node_3->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_3->GetOutDataAnchor(0), node_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_4->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_4->GetOutDataAnchor(1), node_5->GetInDataAnchor(0));
  // construct two isolated node
  ge::OpDescPtr op_6 = std::make_shared<ge::OpDesc>();
  op_6->SetType(REVERSE);
  op_6->SetName("Reverse");
  op_6->AddInputDesc(ge::GeTensorDesc());
  op_6->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_6 = graph->AddNode(op_6);

  ge::OpDescPtr op_7 = std::make_shared<ge::OpDesc>();
  op_7->SetType(REVERSE);
  op_7->SetName("Reverse");
  op_7->AddInputDesc(ge::GeTensorDesc());
  op_7->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node_7 = graph->AddNode(op_7);

  uint64_t size_ori = graph->GetDirectNode().size();

  PrunePass prune_pass;
  vector<GraphPass *> passes = {&prune_pass};
}
// case7: one net path with two DATA node
TEST_F(UtestGraphPassesPrunePass, has_net_out_put_node_with_two_isolate_data_node) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr reverse_op = std::make_shared<ge::OpDesc>();
  reverse_op->SetType(REVERSE);
  reverse_op->SetName("Reverse");
  reverse_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node = graph->AddNode(reverse_op);

  ge::OpDescPtr floor_op = std::make_shared<ge::OpDesc>();
  floor_op->SetType(FLOOR);
  floor_op->SetName("Floor");
  floor_op->AddInputDesc(ge::GeTensorDesc());
  floor_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr floor_node = graph->AddNode(floor_op);

  ge::OpDescPtr net_output_op = std::make_shared<ge::OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  net_output_op->AddInputDesc(ge::GeTensorDesc());
  net_output_op->AddOutputDesc(ge::GeTensorDesc());
  ge::AttrUtils::SetBool(net_output_op, "identity_add_netoutput", true);
  ge::NodePtr netoutput_node = graph->AddNode(net_output_op);
  // construct one isolated DATA node (to be deleted)
  ge::OpDescPtr reverse_op_1 = std::make_shared<ge::OpDesc>();
  reverse_op_1->SetType(REVERSE);
  reverse_op_1->SetName("Reverse1");
  reverse_op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr reverse_node_1 = graph->AddNode(reverse_op_1);

  ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0), floor_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(floor_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  // construct two isolated DATA nodes(to be not deleted)
  ge::OpDescPtr data_op_1 = std::make_shared<ge::OpDesc>();
  data_op_1->SetType(DATA);
  data_op_1->SetName("data");
  data_op_1->AddOutputDesc(ge::GeTensorDesc());
  data_op_1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node_1 = graph->AddNode(data_op_1);

  ge::OpDescPtr data_op_2 = std::make_shared<ge::OpDesc>();
  data_op_2->SetType(DATA);
  data_op_2->SetName("data1");
  data_op_2->AddOutputDesc(ge::GeTensorDesc());
  data_op_2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data_node = graph->AddNode(data_op_2);

  uint64_t size_ori = graph->GetDirectNode().size();
  PrunePass prune_pass;
  std::vector<std::pair<string, GraphPass*>> passes = { {"prune_pass", &prune_pass} };
  Status status = PassManager::Run(graph, passes);

  uint64_t size = graph->GetDirectNode().size();
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(size_ori, (size + 1));

  // it should check net_out_put's input data node and input control node
  auto control_vec = netoutput_node->GetInControlNodes();
  EXPECT_EQ(control_vec.size(), 2);
  // check control_vec contains only data node
  for (auto node : control_vec) {
    bool result = (node->GetName() == "data" || node->GetName() == "data1") ? true : false;
    EXPECT_EQ(result, true);
  }

  auto data_vec = netoutput_node->GetInDataNodes();
  EXPECT_EQ(data_vec.size(), 1);
  // check data_vec contains only Floor node
  for (auto node : data_vec) {
    bool result = (node->GetName() == "Floor") ? true : false;
    EXPECT_EQ(result, true);
  }
}
