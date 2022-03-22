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

#include "common/transop_util.h"

#include "common/debug/log.h"
#include "common/types.h"
#include "common/util.h"
#include "compute_graph.h"

using namespace ge;

class UtestTransopUtil : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestTransopUtil, test_is_transop_true) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", CAST);
  NodePtr node = graph->AddNode(op_desc);

  bool ret = TransOpUtil::IsTransOp(node);
  EXPECT_TRUE(ret);
}

TEST_F(UtestTransopUtil, test_is_transop_fail) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", RELU);
  NodePtr node = graph->AddNode(op_desc);

  bool ret = TransOpUtil::IsTransOp(node);
  EXPECT_FALSE(ret);
}

TEST_F(UtestTransopUtil, test_get_transop_get_index) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr transdata_op_desc = std::make_shared<OpDesc>("Transdata", TRANSDATA);
  OpDescPtr transpose_op_desc = std::make_shared<OpDesc>("Transpose", TRANSPOSE);
  OpDescPtr reshape_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  OpDescPtr cast_op_desc = std::make_shared<OpDesc>("Cast", CAST);

  NodePtr transdata_node = graph->AddNode(transdata_op_desc);
  NodePtr transpose_node = graph->AddNode(transpose_op_desc);
  NodePtr reshape_node = graph->AddNode(reshape_op_desc);
  NodePtr cast_node = graph->AddNode(cast_op_desc);

  int index1 = TransOpUtil::GetTransOpDataIndex(transdata_node);
  int index2 = TransOpUtil::GetTransOpDataIndex(transpose_node);
  int index3 = TransOpUtil::GetTransOpDataIndex(reshape_node);
  int index4 = TransOpUtil::GetTransOpDataIndex(cast_node);

  EXPECT_EQ(index1, 0);
  EXPECT_EQ(index2, 0);
  EXPECT_EQ(index3, 0);
  EXPECT_EQ(index4, 0);
}
