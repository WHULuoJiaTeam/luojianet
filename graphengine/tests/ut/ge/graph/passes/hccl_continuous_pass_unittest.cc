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

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#define protected public
#define private public
#include "graph/passes/hccl_continuous_memcpy_pass.h"
#undef protected
#undef private
#include "graph_builder_utils.h"

namespace ge {
class UtestGraphPassesHcclContinuousMemcpyPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {

/*
 *         var                               var
 *         |  \                             |   \
 *         |   assign                       |   assign
 *         |   //         =======>          |   //
 *     allreduce                         identity
 *        |                                 |
 *       netoutput                        allreduce
 *                                          |
 *                                        netoutput
 */
ComputeGraphPtr BuildGraph_Allreduce_Read_Var_After_Assign(){
  auto builder = ut::GraphBuilder("test");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  return builder.GetGraph();
}
}  // namespace

// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclContinuousMemcpyPass, testInsertIdentityBeforeHccl) {
  ComputeGraphPtr graph = BuildGraph_Allreduce_Read_Var_After_Assign();
  auto src_node = graph->FindNode("var");
  auto dst_node = graph->FindNode("allreduce");
  // test InsertIdentityBeforeHccl
  HcclContinuousMemcpyPass hccl_continuous_memcpy_pass;
  hccl_continuous_memcpy_pass.InsertIdentityBeforeHccl(graph, src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));

  // check
  dst_node = graph->FindNode("allreduce");
  auto in_node_before_dst_node = dst_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(in_node_before_dst_node->GetType(), IDENTITY);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().size(), 1);
  EXPECT_EQ(in_node_before_dst_node->GetInControlNodes().at(0)->GetName(), "assign");
}
}  // namespace ge
