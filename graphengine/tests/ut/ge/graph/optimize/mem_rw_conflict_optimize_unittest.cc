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

#define protected public
#define private public
#include "graph/optimize/graph_optimize.h"
#undef protected
#undef private
#include "../passes/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
class UTest_Graph_Mem_RW_Conflict_Optimize : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
namespace {
/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_Readonly_Subgraph(const string subraph_name){
  auto sub_builder = ut::GraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("data1", DATA, 0,1);
  auto cast = sub_builder.AddNode("cast", CAST, 1,1);
  auto netoutput = sub_builder.AddNode("netoutput",NETOUTPUT, 1,1);
  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX,0);

  sub_builder.AddDataEdge(data1,0,cast,0);
  sub_builder.AddDataEdge(cast,0,netoutput,0);
  return sub_builder.GetGraph();
}
/*
 *      const - allreduce
 *            \ if
 *         insert identity
 */
ComputeGraphPtr BuildGraph_Readonly_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto ctrl_const = builder.AddNode("ctrl_const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1,0);

  builder.AddDataEdge(const1, 0, allreduce, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);
  builder.AddControlEdge(ctrl_const, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*       const1---allreduce  const1--identity - allreduce
 *               /                 /
 *  var-identity--cast1   ==>   var-----cast1
 *              \                 \
 *               if                if
 */
ComputeGraphPtr BuildGraph_Identiyt_Split(){
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto identity = builder.AddNode("identity", IDENTITY, 1, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 1, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto if_node = builder.AddNode("if", IF, 1,0);

  builder.AddDataEdge(var, 0 , identity, 0);
  builder.AddDataEdge(identity, 0 , allreduce, 0);
  builder.AddDataEdge(identity, 0 , cast1, 0);
  builder.AddDataEdge(identity, 0 , if_node, 0);
  builder.AddControlEdge(const1, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}
/*
 * mul == allreduce
 * need insert identity
 */
ComputeGraphPtr BuildGraph_mul_1To2_ScopeWrite() {
  auto builder = ut::GraphBuilder("test");
  auto mul = builder.AddNode("mul", MUL, 2,1);
  auto allreduce = builder.AddNode("allreduce", HCOMALLREDUCE, 2,0);
  AttrUtils::SetBool(allreduce->GetOpDesc(), "_input_mutable", true);
  builder.AddDataEdge(mul,0,allreduce,0);
  builder.AddDataEdge(mul,0,allreduce,1);
  return builder.GetGraph();
}
}  // namespace
// const -> allreduce
// const -> Identity -> allreduce
TEST(UtestGraphPassesHcclMemcpyPass, testReadonlyScopeWriteConflict) {
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  EXPECT_EQ(allreduce->GetInDataNodes().at(0)->GetType(), IDENTITY);
}
TEST(UtestGraphPassesHcclMemcpyPass, testIdentiytSplit) {
  ComputeGraphPtr graph = BuildGraph_Identiyt_Split();
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto allreduce = graph->FindNode("allreduce");
  auto allreduce_in_node = allreduce->GetInDataNodes().at(0);
  EXPECT_EQ(allreduce_in_node->GetType(), IDENTITY);
  EXPECT_EQ(allreduce_in_node->GetInControlNodes().at(0)->GetType(), CONSTANT);
}
TEST(UtestGraphPassesHcclMemcpyPass, testMul_1To2_ScopeWrite) {
  ComputeGraphPtr graph = BuildGraph_mul_1To2_ScopeWrite();
  EXPECT_EQ(graph->GetDirectNodesSize(), 2);
  GraphOptimize graph_optimizer;
  auto ret = graph_optimizer.HandleMemoryRWConflict(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
}
}  // namespace ge
