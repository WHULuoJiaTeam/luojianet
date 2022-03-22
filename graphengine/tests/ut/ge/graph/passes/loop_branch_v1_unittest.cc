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

#include "graph/passes/merge_input_memcpy_pass.h"
#include "graph/passes/next_iteration_pass.h"
#include "graph/passes/switch_to_stream_switch_pass.h"
#include "graph/passes/merge_to_stream_merge_pass.h"
#include "graph/passes/attach_stream_label_pass.h"

#include <gtest/gtest.h>
#include "graph_builder_utils.h"

namespace ge {
class UtestLoopBranchV1Pass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
///
/// net_output
///    |
///   exit      next_iteration
///     \        |          |
///      \      add         |
///      F\   T/   \        |
///       switch1  enter1   |
///      /     |     |      |
/// loop_cond  |   const1   |
///      |     |            |
///     less   |            |
///    /    \  |            |
/// enter2   merge ---------|
///   |        |
/// const2   enter3
///            |
///           var
///
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto enter1 = builder.AddNode("enter1", ENTER, 1, 1);
  AttrUtils::SetStr(enter1->GetOpDesc(), ENTER_ATTR_FRAME_NAME, "frame_name");
  auto const2 = builder.AddNode("const2", CONSTANTOP, 0, 1);
  auto enter2 = builder.AddNode("enter2", ENTER, 1, 1);
  AttrUtils::SetStr(enter2->GetOpDesc(), ENTER_ATTR_FRAME_NAME, "frame_name");
  auto var = builder.AddNode("var", VARIABLEV2, 0, 1);
  auto enter3 = builder.AddNode("enter3", ENTER, 1, 1);
  AttrUtils::SetStr(enter3->GetOpDesc(), ENTER_ATTR_FRAME_NAME, "frame_name");
  auto merge = builder.AddNode("merge", MERGE, 2, 2);
  auto less = builder.AddNode("less", LESS, 2, 1);
  auto loop_cond = builder.AddNode("loop_cond", LOOPCOND, 1, 1, FORMAT_ND, DT_BOOL, {});
  auto switch1 = builder.AddNode("switch1", SWITCH, 2, 2);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto next_iteration = builder.AddNode("next_iteration", NEXTITERATION, 1, 1);
  auto exit = builder.AddNode("exit", EXIT, 1, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, enter1, 0);
  builder.AddDataEdge(const2, 0, enter2, 0);
  builder.AddDataEdge(var, 0, enter3, 0);
  builder.AddDataEdge(enter3, 0, merge, 0);
  builder.AddDataEdge(enter2, 0, less, 0);
  builder.AddDataEdge(merge, 0, less, 1);
  builder.AddDataEdge(merge, 0, switch1, 0);
  builder.AddDataEdge(less, 0, loop_cond, 0);
  builder.AddDataEdge(loop_cond, 0, switch1, 1);
  builder.AddDataEdge(switch1, 1, add, 0);
  builder.AddDataEdge(enter1, 0, add, 1);
  builder.AddDataEdge(add, 0, next_iteration, 0);
  builder.AddDataEdge(next_iteration, 0, merge, 1);
  builder.AddDataEdge(switch1, 0, exit, 0);
  builder.AddDataEdge(exit, 0, net_output, 0);
  return builder.GetGraph();
}
}  // namespace

TEST_F(UtestLoopBranchV1Pass, common_loop_branch_v1) {
  auto graph = BuildGraph1();
  MergeInputMemcpyPass memcpy_pass;
  NextIterationPass loop_pass;
  SwitchToStreamSwitchPass switch_pass;
  MergeToStreamMergePass merge_pass;
  AttachStreamLabelPass label_pass;
  EXPECT_EQ(memcpy_pass.Run(graph), SUCCESS);
  EXPECT_EQ(loop_pass.Run(graph), SUCCESS);
  EXPECT_EQ(switch_pass.Run(graph), SUCCESS);
  EXPECT_EQ(merge_pass.Run(graph), SUCCESS);
  EXPECT_EQ(label_pass.Run(graph), SUCCESS);

  uint32_t switch_num = 0;
  uint32_t merge_num = 0;
  uint32_t cast_num = 0;
  uint32_t stream_switch_num = 0;
  uint32_t active_num = 0;
  uint32_t stream_merge_num = 0;
  uint32_t memcpy_num = 0;

  for (const auto &node : graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    std::string type = op_desc->GetType();
    if (type == SWITCH || type == REFSWITCH) {
      switch_num++;
    } else if (type == MERGE) {
      merge_num++;
    } else if (type == CAST) {
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_STREAM_LABEL));
      cast_num++;
    } else if (type == STREAMSWITCH) {
      stream_switch_num++;
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_STREAM_LABEL));
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_ACTIVE_LABEL_LIST));
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE));
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG));
    } else if (type == STREAMMERGE) {
      stream_merge_num++;
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_STREAM_LABEL));
    } else if (type == STREAMACTIVE) {
      active_num++;
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_ACTIVE_LABEL_LIST));
    } else if (type == MEMCPYASYNC) {
      memcpy_num++;
    }
  }

  EXPECT_EQ(switch_num, 0);
  EXPECT_EQ(merge_num, 0);
  EXPECT_EQ(cast_num, 1);
  EXPECT_EQ(stream_switch_num, 2);
  EXPECT_EQ(active_num, 3);
  EXPECT_EQ(stream_merge_num, 1);
  EXPECT_EQ(memcpy_num, 0);
}

}  // namespace ge
