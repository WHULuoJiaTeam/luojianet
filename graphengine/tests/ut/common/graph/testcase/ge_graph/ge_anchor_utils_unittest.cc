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
#include <iostream>

#define protected public
#include "graph/utils/anchor_utils.h"

#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#undef protected

using namespace ge;

class UtestGeAnchorUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeAnchorUtils, base) {
  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  InDataAnchorPtr a1 = std::make_shared<InDataAnchor>(n1, 0);

  EXPECT_EQ(AnchorUtils::SetFormat(a1, FORMAT_ND), GRAPH_SUCCESS);
  Format f1 = AnchorUtils::GetFormat(a1);
  EXPECT_EQ(f1, FORMAT_ND);

  InDataAnchorPtr a2 = std::make_shared<InDataAnchor>(n1, 0);
  EXPECT_EQ(AnchorUtils::SetFormat(nullptr, FORMAT_ND), GRAPH_FAILED);
  Format f2 = AnchorUtils::GetFormat(nullptr);
  EXPECT_EQ(f2, FORMAT_RESERVED);

  // has control edge
  OpDescPtr desc_ptr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr1 = std::make_shared<ComputeGraph>("name");
  n1 = graph_ptr1->AddNode(desc_ptr1);
  NodePtr n2 = graph_ptr1->AddNode(desc_ptr2);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::HasControlEdge(n1->GetOutControlAnchor()), true);
  EXPECT_EQ(AnchorUtils::IsControlEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor()), true);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutControlAnchor(), n2->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::HasControlEdge(n1->GetOutDataAnchor(0)), true);
  EXPECT_EQ(AnchorUtils::IsControlEdge(n1->GetOutDataAnchor(0), n2->GetInControlAnchor()), true);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutDataAnchor(0), n2->GetInControlAnchor()), GRAPH_SUCCESS);

  EXPECT_EQ(GraphUtils::AddEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::HasControlEdge(n1->GetOutDataAnchor(0)), false);
  EXPECT_EQ(AnchorUtils::IsControlEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), false);
  EXPECT_EQ(GraphUtils::RemoveEdge(n1->GetOutDataAnchor(0), n2->GetInDataAnchor(0)), GRAPH_SUCCESS);
}
