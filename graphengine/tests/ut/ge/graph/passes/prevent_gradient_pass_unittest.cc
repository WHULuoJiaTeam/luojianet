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
#include "graph/passes/prevent_gradient_pass.h"

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

using namespace std;
using namespace testing;
using namespace ge;

class UtestPreventGradientPass : public Test {
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

TEST_F(UtestPreventGradientPass, succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = AddNode(graph, "PreventGradient", PREVENTGRADIENT);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);

  GraphUtils::AddEdge(node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));

  PreventGradientPass pass;
  Status status = pass.Run(node);
  EXPECT_EQ(status, SUCCESS);
  NodePtr found_node = graph->FindNode("PreventGradient");
  EXPECT_EQ(found_node, nullptr);

  status = pass.Run(reduce_min_node);
  EXPECT_EQ(status, SUCCESS);

  string type2 = "FrameworkOp";
  node->GetOpDesc()->SetType(type2);
  status = pass.Run(node);
}
