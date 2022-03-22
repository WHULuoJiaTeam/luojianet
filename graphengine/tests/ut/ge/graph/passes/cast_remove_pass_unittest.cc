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

#define protected public
#define private public
#include "graph/passes/cast_remove_pass.h"
#undef protected
#undef private

#include "anchor.h"
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/pass_manager.h"
#include "graph_builder_utils.h"
#include <string>
#include <iostream>
#include <vector>
#include "opskernel_manager/ops_kernel_manager.h"
#include "omg/omg_inner_types.h"


using namespace testing;
using namespace ge;
using namespace std;

class UtestGraphPassesCastRemovePass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// case1:no net_out_put_node
// TEST_F(UtestGraphPassesCastRemovePass, DoFuseProcess) {
//   std::vector<NodePtr> nodes_to_fuse;

//   auto builder = ut::GraphBuilder("g1");
//   auto data = builder.AddNode("data", DATA, 1, 1);
//   auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
//   cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
//   auto trans = builder.AddNode("trans", TRANSPOSE, 1, 1, FORMAT_NCHW, DT_FLOAT16);
//   auto cast2 = builder.AddNode("cast2", CAST, 1, 1);
//   cast2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
//   auto net = builder.AddNode("netout", NETOUTPUT, 1, 1);

//   builder.AddDataEdge(data, 0, cast1, 0);
//   builder.AddDataEdge(cast1, 0, trans, 0);
//   builder.AddDataEdge(trans, 0, cast2, 0);
//   builder.AddDataEdge(cast2, 0, net, 0);
//   ComputeGraphPtr compute_graph = builder.GetGraph();

//   map<string, string> options;

//   CastRemovePass cast_remove_pass;
//   DataType type = DT_FLOAT;
//   nodes_to_fuse.emplace_back(cast1);
//   nodes_to_fuse.emplace_back(trans);
//   nodes_to_fuse.emplace_back(cast2);
//   OpsKernelManager ops_kernel_manager;
//   cast_remove_pass.DoFuse(ops_kernel_manager, type, nodes_to_fuse);
//   EXPECT_EQ(compute_graph->GetAllNodesSize(),5);
//   std::vector<size_t> to_be_deleted_cast_index;
//   to_be_deleted_cast_index.emplace_back(0);
//   to_be_deleted_cast_index.emplace_back(2);
//   (void)cast_remove_pass.DoRemoveCast(to_be_deleted_cast_index, nodes_to_fuse);
//   EXPECT_EQ(compute_graph->GetAllNodesSize(),3);
// }

TEST_F(UtestGraphPassesCastRemovePass, DoFuseProcess) {
  std::vector<NodePtr> nodes_to_fuse;

  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data", DATA, 1, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  auto trans = builder.AddNode("trans", TRANSPOSE, 1, 1, FORMAT_NCHW, DT_FLOAT16);
  auto cast2 = builder.AddNode("cast2", CAST, 1, 1);
  cast2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  auto net = builder.AddNode("netout", NETOUTPUT, 1, 1);

  builder.AddDataEdge(data, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, trans, 0);
  builder.AddDataEdge(trans, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, net, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();

  map<string, string> options;

  CastRemovePass cast_remove_pass;
  DataType type = DT_FLOAT;
  nodes_to_fuse.emplace_back(cast1);
  nodes_to_fuse.emplace_back(trans);
  nodes_to_fuse.emplace_back(cast2);
  cast_remove_pass.RemoveCast(type, nodes_to_fuse);
  EXPECT_EQ(compute_graph->GetAllNodesSize(),3);
}
