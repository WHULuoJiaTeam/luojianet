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
#include "graph/passes/switch_to_stream_switch_pass.h"
#include "graph/passes/merge_to_stream_merge_pass.h"
#include "graph/passes/attach_stream_label_pass.h"

#include <gtest/gtest.h>
#include "graph_builder_utils.h"

namespace ge {
class UtestCondBranchV1Pass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
///
///    net_output
///         |
///       merge
///       /   \.
/// square     add
///   F|   T/   T\.
///   switch1     switch2
///  /       \   /       \.
/// var1     var2       var3
///
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", VARIABLEV2, 0, 1);
  auto var2 = builder.AddNode("var2", VARIABLEV2, 0, 1, FORMAT_ND, DT_BOOL, {});
  auto var3 = builder.AddNode("var3", VARIABLEV2, 0, 1);
  auto switch1 = builder.AddNode("switch1", REFSWITCH, 2, 2);
  auto switch2 = builder.AddNode("switch2", SWITCH, 2, 2);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto square = builder.AddNode("square", SQUARE, 1, 1);
  auto merge = builder.AddNode("merge", MERGE, 2, 2);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(var2, 0, switch1, 1);
  builder.AddDataEdge(var3, 0, switch2, 0);
  builder.AddDataEdge(var2, 0, switch2, 1);
  builder.AddDataEdge(switch1, 0, square, 0);
  builder.AddDataEdge(switch1, 1, add, 0);
  builder.AddDataEdge(switch2, 1, add, 1);
  builder.AddDataEdge(square, 0, merge, 0);
  builder.AddDataEdge(add, 0, merge, 1);
  builder.AddDataEdge(merge, 0, net_output, 0);
  return builder.GetGraph();
}
}  // namespace

TEST_F(UtestCondBranchV1Pass, common_cond_branch_v1) {
  auto graph = BuildGraph1();
  MergeInputMemcpyPass memcpy_pass;
  SwitchToStreamSwitchPass switch_pass;
  MergeToStreamMergePass merge_pass;
  AttachStreamLabelPass label_pass;
  EXPECT_EQ(memcpy_pass.Run(graph), SUCCESS);
  EXPECT_EQ(switch_pass.Run(graph), SUCCESS);
  EXPECT_EQ(merge_pass.Run(graph), SUCCESS);
  EXPECT_EQ(label_pass.Run(graph), SUCCESS);

  uint32_t switch_num = 0;
  uint32_t merge_num = 0;
  uint32_t cast_num = 0;
  uint32_t stream_switch_num = 0;
  uint32_t memcpy_num = 0;
  uint32_t active_num = 0;
  uint32_t stream_merge_num = 0;

  for (const auto &node : graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    std::string type = op_desc->GetType();
    if (type == SWITCH || type == REFSWITCH) {
      switch_num++;
    } else if (type == MERGE) {
      merge_num++;
    } else if (type == CAST) {
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
    } else if ((type == MEMCPYASYNC) || (type == MEMCPYADDRASYNC)) {
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_STREAM_LABEL));
      memcpy_num++;
    } else if (type == STREAMACTIVE) {
      active_num++;
      EXPECT_TRUE(op_desc->HasAttr(ATTR_NAME_ACTIVE_LABEL_LIST));
    }
  }

  EXPECT_EQ(switch_num, 0);
  EXPECT_EQ(merge_num, 0);
  EXPECT_EQ(cast_num, 1);
  EXPECT_EQ(stream_switch_num, 2);
  EXPECT_EQ(memcpy_num, 2);
  EXPECT_EQ(active_num, 3);
  EXPECT_EQ(stream_merge_num, 1);
}

}  // namespace ge
