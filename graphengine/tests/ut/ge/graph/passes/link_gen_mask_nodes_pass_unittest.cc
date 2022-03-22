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

#include "graph/passes/link_gen_mask_nodes_pass.h"


#include <gtest/gtest.h>
#include <set>
#include <string>

#include "graph_builder_utils.h"

namespace ge {
class UtestLinkGenMaskNodesPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
  /*
   *   do_mask1       do_mask2    do_mask3       do_mask4     do_mask5      do_mask6
   *      \| \          / |/        |/ \          /  |/         \| \          / |/
   *       \  \        /  |         |   \        /   |           |  \        /  |
   *        \  genmask1   |         |    genmask2    |           |   genmask3   |
   *         \     | |    |         |      |  |      |           |     | |     /
   *          ----------------------const1 and const2--------------------------
   */
ut::GraphBuilder Graph1Builder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto gen_mask1 = builder.AddNode("gen_mask1_DropOutGenMask", "DropOutGenMask", 2, 1);
  auto gen_mask2 = builder.AddNode("gen_mask2", "DropOutGenMaskV3", 2, 1);
  auto gen_mask3 = builder.AddNode("gen_mask3", "DropOutGenMaskV3D", 2, 1);
  auto do_mask1 = builder.AddNode("do_mask1", "DropOutDoMask", 3, 1);
  auto do_mask2 = builder.AddNode("do_mask2", "DropOutDoMask", 3, 1);
  auto do_mask3 = builder.AddNode("do_mask3", "DropOutDoMask", 3, 1);
  auto do_mask4 = builder.AddNode("do_mask4", "DropOutDoMask", 3, 1);
  auto do_mask5 = builder.AddNode("do_mask5", "DropOutDoMask", 3, 1);
  auto do_mask6 = builder.AddNode("do_mask6", "DropOutDoMask", 3, 1);
  gen_mask1->GetOpDesc()->SetOpEngineName("DNN_HCCL");
  gen_mask2->GetOpDesc()->SetOpEngineName("DNN_HCCL");
  gen_mask3->GetOpDesc()->SetOpEngineName("DNN_HCCL");

  builder.AddDataEdge(const1, 0, gen_mask1, 0);
  builder.AddDataEdge(const1, 0, gen_mask2, 0);
  builder.AddDataEdge(const1, 0, gen_mask3, 0);
  builder.AddDataEdge(const1, 0, do_mask1, 0);
  builder.AddDataEdge(const1, 0, do_mask2, 0);
  builder.AddDataEdge(const1, 0, do_mask3, 0);
  builder.AddDataEdge(const1, 0, do_mask4, 0);
  builder.AddDataEdge(const1, 0, do_mask5, 0);
  builder.AddDataEdge(const1, 0, do_mask6, 0);
  builder.AddDataEdge(gen_mask1, 0, do_mask1, 1);
  builder.AddDataEdge(gen_mask1, 0, do_mask2, 1);
  builder.AddDataEdge(gen_mask2, 0, do_mask3, 1);
  builder.AddDataEdge(gen_mask2, 0, do_mask4, 1);
  builder.AddDataEdge(gen_mask3, 0, do_mask5, 1);
  builder.AddDataEdge(gen_mask3, 0, do_mask6, 1);
  builder.AddDataEdge(const2, 0, gen_mask1, 1);
  builder.AddDataEdge(const2, 0, gen_mask2, 1);
  builder.AddDataEdge(const2, 0, gen_mask3, 1);
  builder.AddDataEdge(const2, 0, do_mask1, 2);
  builder.AddDataEdge(const2, 0, do_mask2, 2);
  builder.AddDataEdge(const2, 0, do_mask3, 2);
  builder.AddDataEdge(const2, 0, do_mask4, 2);
  builder.AddDataEdge(const2, 0, do_mask5, 2);
  builder.AddDataEdge(const2, 0, do_mask6, 2);
  return builder;
}
}  // namespace


TEST_F(UtestLinkGenMaskNodesPass, link_gen_mask_nodes_pass_success) {
  auto builder = Graph1Builder();
  auto graph = builder.GetGraph();
  
  std::map<std::string, int> stream_max_parallel_num;
  stream_max_parallel_num["DNN_HCCL"] = 1;
  LinkGenMaskNodesPass link_pass(stream_max_parallel_num);
  Status ret = link_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  auto gen_mask2 = graph->FindNode("gen_mask2");
  EXPECT_NE(gen_mask2, nullptr);

  auto in_ctrl_nodes = gen_mask2->GetInControlNodes();
  EXPECT_EQ(in_ctrl_nodes.size(), 1);
  auto in_ctrl_node = in_ctrl_nodes.at(0);
  EXPECT_EQ(in_ctrl_node->GetName(), "gen_mask3");

  auto out_ctrl_nodes = gen_mask2->GetOutControlNodes();
  EXPECT_EQ(out_ctrl_nodes.size(), 1);
  auto out_ctrl_node = out_ctrl_nodes.at(0);
  EXPECT_EQ(out_ctrl_node->GetName(), "gen_mask1_DropOutGenMask");
}
}  // namespace ge
