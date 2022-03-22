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

#include "graph/passes/shape_operate_op_remove_pass.h"

#include <gtest/gtest.h>

using namespace ge;

class UtestGraphPassesShapeOperateOpRemovePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 2,
                  int32_t out_anchors_num = 2) {
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
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

TEST_F(UtestGraphPassesShapeOperateOpRemovePass, squeeze_and_squeeze) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  NodePtr transpose_node = AddNode(graph, "transpose1", PERMUTE);
  NodePtr squeeze_node = AddNode(graph, "squeeze1", SQUEEZE);

  GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), squeeze_node->GetInDataAnchor(0));

  ge::ShapeOperateOpRemovePass shape_operate_op_pass;
  Status status = shape_operate_op_pass.Run(graph);
  EXPECT_EQ(SUCCESS, status);
  NodePtr found_node = graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, found_node);
}
