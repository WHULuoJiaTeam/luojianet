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

#define protected public
#define private public
#include "graph/passes/identity_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_builder_utils.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;

class UtestIdentityPass : public Test {
 protected:
  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
    GeTensorDesc tensor_desc;
    OpDescPtr opdesc = make_shared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(opdesc);
    return node;
  }
};

///  merge1
///    |
/// identity1
///   |   \c
/// var1  var2
static ComputeGraphPtr BuildGraph1() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 1);

  builder.AddDataEdge(var1, 0, identity1, 0);
  builder.AddControlEdge(var2, identity1);
  builder.AddDataEdge(identity1, 0, merge1, 0);
  return builder.GetGraph();
}

///   addn1
///    |c
///  identity1
///    |
///  switch1
///    |
///   var1
static ComputeGraphPtr BuildGraph2() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, identity1, 0);
  builder.AddControlEdge(identity1, addn1);
  return builder.GetGraph();
}

///  addn1
///    |
/// identity1
///    |
/// switch1
///    |
///  var1
static ComputeGraphPtr BuildGraph3() {
  ge::ut::GraphBuilder builder("g3");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, identity1, 0);
  builder.AddDataEdge(identity1, 0, addn1, 0);
  return builder.GetGraph();
}

TEST_F(UtestIdentityPass, succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = AddNode(graph, "Identity", IDENTITY);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);

  GraphUtils::AddEdge(node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));

  IdentityPass pass(true);
  Status status = pass.Run(node);
  EXPECT_EQ(status, SUCCESS);
  NodePtr found_node = graph->FindNode("Identity");
  EXPECT_EQ(found_node, nullptr);

  status = pass.Run(reduce_min_node);
  EXPECT_EQ(status, SUCCESS);

  string type2 = "FrameworkOp";
  node->GetOpDesc()->SetType(type2);
  status = pass.Run(node);

  NodePtr node_err = AddNode(graph, "Identity", IDENTITY, 1, 2);
  status = pass.Run(node_err);
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestIdentityPass, skip_merge) {
  auto graph = BuildGraph1();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_NE(identity1, nullptr);
  EXPECT_EQ(identity1->GetOutNodes().size(), 1);
  EXPECT_EQ(identity1->GetOutDataNodes().at(0)->GetName(), "merge1");
  EXPECT_EQ(identity1->GetInNodes().size(), 2);

  names_to_pass.clear();
  IdentityPass force_pass(true);
  names_to_pass.emplace_back("ForceIdentityPass", &force_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
}

TEST_F(UtestIdentityPass, skip_switch) {
  auto graph = BuildGraph2();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_NE(identity1, nullptr);
  EXPECT_EQ(identity1->GetInNodes().size(), 1);
  EXPECT_EQ(identity1->GetInDataNodes().at(0)->GetName(), "switch1");

  names_to_pass.clear();
  IdentityPass force_pass(true);
  names_to_pass.emplace_back("ForceIdentityPass", &force_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
}

TEST_F(UtestIdentityPass, norm_after_switch) {
  auto graph = BuildGraph3();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 1);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "addn1");
}