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
#include "graph/passes/assert_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace testing;
namespace ge {
class UtestGraphPassesAssertPass : public Test {
 protected:
  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
    GeTensorDesc tensor_desc;
    OpDescPtr op_desc = make_shared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      op_desc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      op_desc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(op_desc);
    return node;
  }
};

///             D      E
///           |   \  |  \.
///          F     C     G
///             :  |  :
///            H    A   I
///                :
///                B
TEST_F(UtestGraphPassesAssertPass, assert_pass_test1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  NodePtr node_a = AddNode(graph, "A", ge::ASSERT, 1, 0);
  NodePtr node_b = AddNode(graph, "B", "B", 1, 1);
  NodePtr node_c = AddNode(graph, "C", "C", 2, 1);
  NodePtr node_d = AddNode(graph, "D", "D", 1, 2);
  NodePtr node_e = AddNode(graph, "E", "E", 1, 2);
  NodePtr node_f = AddNode(graph, "F", "F", 1, 1);
  NodePtr node_g = AddNode(graph, "G", "G", 1, 1);
  NodePtr node_h = AddNode(graph, "H", "H", 1, 1);
  NodePtr node_i = AddNode(graph, "I", "I", 1, 1);
  GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_b->GetInControlAnchor());
  GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_c->GetOutControlAnchor(), node_h->GetInControlAnchor());
  GraphUtils::AddEdge(node_c->GetOutControlAnchor(), node_i->GetInControlAnchor());
  GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_d->GetOutDataAnchor(1), node_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_c->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_e->GetOutDataAnchor(1), node_g->GetInDataAnchor(0));

  AssertPass assert_pass;
  Status status = assert_pass.Run(node_a);
  EXPECT_EQ(SUCCESS, status);

  EXPECT_EQ(node_d->GetOutControlNodes().size(), 3);
  EXPECT_EQ(node_e->GetOutControlNodes().size(), 3);
  EXPECT_EQ(node_h->GetInControlNodes().size(), 2);
  EXPECT_EQ(node_b->GetInControlNodes().size(), 2);
  EXPECT_EQ(node_i->GetInControlNodes().size(), 2);

  EXPECT_EQ(graph->FindNode("A"), nullptr);
  EXPECT_EQ(graph->FindNode("C"), nullptr);
}

///                   G      E
///                  | \   :
///           C     G  D
///               :    |   :
///                    A     F
///                    :
///                    B
TEST_F(UtestGraphPassesAssertPass, assert_pass_test2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  NodePtr node_a = AddNode(graph, "A", ge::ASSERT, 1, 0);
  NodePtr node_b = AddNode(graph, "B", "B", 1, 1);
  NodePtr node_c = AddNode(graph, "C", "C", 1, 1);
  NodePtr node_d = AddNode(graph, "D", "D", 1, 1);
  NodePtr node_e = AddNode(graph, "E", "E", 1, 1);
  NodePtr node_f = AddNode(graph, "F", "F", 1, 1);
  NodePtr node_g = AddNode(graph, "G", "G", 1, 2);
  NodePtr node_h = AddNode(graph, "H", "H", 1, 1);
  GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_b->GetInControlAnchor());
  GraphUtils::AddEdge(node_c->GetOutControlAnchor(), node_a->GetInControlAnchor());
  GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_d->GetOutControlAnchor(), node_f->GetInControlAnchor());
  GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_g->GetOutDataAnchor(1), node_h->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_e->GetOutControlAnchor(), node_d->GetInControlAnchor());

  AssertPass assert_pass;
  Status status = assert_pass.Run(node_a);
  EXPECT_EQ(SUCCESS, status);

  EXPECT_EQ(node_g->GetOutControlNodes().size(), 2);
  EXPECT_EQ(node_c->GetOutControlAnchor()->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(node_c->GetOutControlAnchor()->GetPeerInControlAnchors().at(0), node_b->GetInControlAnchor());

  EXPECT_EQ(node_e->GetOutControlNodes().size(), 2);

  EXPECT_EQ(graph->FindNode("A"), nullptr);
  EXPECT_EQ(graph->FindNode("D"), nullptr);
}

///     E         F
///    | \       | \.
///   H    C ->  D   G
///        \   |   :
///           A      I
///           :
///           B
TEST_F(UtestGraphPassesAssertPass, assert_pass_test3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  NodePtr node_a = AddNode(graph, "A", ge::ASSERT, 2, 0);
  NodePtr node_b = AddNode(graph, "B", "B", 1, 1);
  NodePtr node_c = AddNode(graph, "C", "C", 1, 2);
  NodePtr node_d = AddNode(graph, "D", "D", 2, 1);
  NodePtr node_e = AddNode(graph, "E", "E", 1, 2);
  NodePtr node_f = AddNode(graph, "F", "F", 1, 2);
  NodePtr node_g = AddNode(graph, "G", "G", 1, 1);
  NodePtr node_h = AddNode(graph, "H", "H", 1, 1);
  NodePtr node_i = AddNode(graph, "I", "I", 1, 1);
  GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_b->GetInControlAnchor());
  GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_c->GetOutDataAnchor(1), node_d->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_a->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_d->GetOutControlAnchor(), node_i->GetInControlAnchor());
  GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_e->GetOutDataAnchor(1), node_h->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_f->GetOutDataAnchor(1), node_g->GetInDataAnchor(0));

  AssertPass assert_pass;
  Status status = assert_pass.Run(node_a);
  EXPECT_EQ(SUCCESS, status);

  EXPECT_EQ(node_e->GetOutControlNodes().size(), 2);
  EXPECT_EQ(node_f->GetOutControlNodes().size(), 2);
  EXPECT_EQ(node_b->GetInControlNodes().size(), 2);
  EXPECT_EQ(node_i->GetInControlNodes().size(), 2);

  EXPECT_EQ(graph->FindNode("A"), nullptr);
  EXPECT_EQ(graph->FindNode("C"), nullptr);
  EXPECT_EQ(graph->FindNode("D"), nullptr);
}
}  // namespace ge
