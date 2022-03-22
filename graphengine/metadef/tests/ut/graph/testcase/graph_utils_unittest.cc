/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_local_context.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"

#undef private
#undef protected

namespace ge {
namespace {
bool IfNodeExist(const ComputeGraphPtr &graph, std::function<bool(const NodePtr &)> filter,
                 bool direct_node_flag = true) {
  for (const auto &node : graph->GetNodes(direct_node_flag)) {
    if (filter(node)) {
      return true;
    }
  }
  return false;
}
/*
 *             data1  const1         data2  const2
 *                \    /                \    /
 *                 add1                  add2
 *                   |                    |
 *                 cast1                cast2
 *                   |                    |
 *                square1  var1  var2  square2
 *                     \   /  |  |  \   /
 *                     less1  |  |  less2
 *                          \ |  | /
 *                            mul
 *                             |
 *                             |
 *                             |
 *                          netoutput
 */
void BuildGraphForUnfold(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &var1 = builder.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &input2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &var2 = builder.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 4, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input1, 0, func, 0);
  builder.AddDataEdge(var1, 0, func, 1);
  builder.AddDataEdge(input2, 0, func, 2);
  builder.AddDataEdge(var2, 0, func, 3);
  builder.AddDataEdge(func, 0, netoutput, 0);

  graph = builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &const1 = sub_builder.AddNode("const1", CONSTANTOP, 0, 1);
  const auto &add1 = sub_builder.AddNode("add1", "Add", 2, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &func1 = sub_builder.AddNode("func1", PARTITIONEDCALL, 2, 1);
  const auto &data2 = sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = sub_builder.AddNode("data3", DATA, 1, 1);
  const auto &const2 = sub_builder.AddNode("const2", CONSTANTOP, 0, 1);
  const auto &add2 = sub_builder.AddNode("add2", "Add", 2, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &func2 = sub_builder.AddNode("func2", PARTITIONEDCALL, 2, 1);
  const auto &data4 = sub_builder.AddNode("data4", DATA, 1, 1);
  const auto &mul = sub_builder.AddNode("mul", "Mul", 2, 1);
  const auto &netoutput0 = sub_builder.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(data1, 0, add1, 0);
  sub_builder.AddDataEdge(const1, 0, add1, 1);
  sub_builder.AddDataEdge(add1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, func1, 0);
  sub_builder.AddDataEdge(data2, 0, func1, 1);
  sub_builder.AddDataEdge(data3, 0, add2, 0);
  sub_builder.AddDataEdge(const2, 0, add2, 1);
  sub_builder.AddDataEdge(add2, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, func2, 0);
  sub_builder.AddDataEdge(data4, 0, func2, 1);
  sub_builder.AddDataEdge(func1, 0, mul, 0);
  sub_builder.AddDataEdge(func2, 0, mul, 1);
  sub_builder.AddDataEdge(mul, 0, netoutput0, 0);

  subgraph = sub_builder.GetGraph();
  subgraph->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
  AttrUtils::SetInt(data4->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 3);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);

  auto sub_sub_builder1 = ut::GraphBuilder("sub_sub1");
  const auto &data5 = sub_sub_builder1.AddNode("data5", DATA, 1, 1);
  const auto &data6 = sub_sub_builder1.AddNode("data6", DATA, 1, 1);
  const auto &square1 = sub_sub_builder1.AddNode("square1", "Square", 1, 1);
  const auto &less1 = sub_sub_builder1.AddNode("less1", "Less", 2, 1);
  const auto &netoutput1 = sub_sub_builder1.AddNode("netoutput1", NETOUTPUT, 1, 0);
  sub_sub_builder1.AddDataEdge(data5, 0, square1, 0);
  sub_sub_builder1.AddDataEdge(square1, 0, less1, 0);
  sub_sub_builder1.AddDataEdge(data6, 0, less1, 1);
  sub_sub_builder1.AddDataEdge(less1, 0, netoutput1, 0);

  const auto &sub_subgraph1 = sub_sub_builder1.GetGraph();
  sub_subgraph1->SetGraphUnknownFlag(true);
  AttrUtils::SetInt(data5->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data6->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func1->GetOpDesc()->AddSubgraphName("f");
  func1->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph1->GetName());
  graph->AddSubGraph(sub_subgraph1);
  sub_subgraph1->SetParentNode(func1);
  sub_subgraph1->SetParentGraph(subgraph);

  auto sub_sub_builder2 = ut::GraphBuilder("sub_sub2");
  const auto &data7 = sub_sub_builder2.AddNode("data7", DATA, 1, 1);
  const auto &data8 = sub_sub_builder2.AddNode("data8", DATA, 1, 1);
  const auto &square2 = sub_sub_builder2.AddNode("square2", "Square", 1, 1);
  const auto &less2 = sub_sub_builder2.AddNode("less2", "Less", 2, 1);
  const auto &netoutput2 = sub_sub_builder2.AddNode("netoutput2", NETOUTPUT, 1, 0);
  sub_sub_builder2.AddDataEdge(data7, 0, square2, 0);
  sub_sub_builder2.AddDataEdge(square2, 0, less2, 0);
  sub_sub_builder2.AddDataEdge(data8, 0, less2, 1);
  sub_sub_builder2.AddDataEdge(less2, 0, netoutput2, 0);

  const auto &sub_subgraph2 = sub_sub_builder2.GetGraph();
  sub_subgraph2->SetGraphUnknownFlag(false);
  AttrUtils::SetInt(data7->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(data8->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  func2->GetOpDesc()->AddSubgraphName("f");
  func2->GetOpDesc()->SetSubgraphInstanceName(0, sub_subgraph2->GetName());
  graph->AddSubGraph(sub_subgraph2);
  sub_subgraph2->SetParentNode(func2);
  sub_subgraph2->SetParentGraph(subgraph);

  return;
}

ComputeGraphPtr BuildGraphWithSubGraph() {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("data1", "Data", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &data2 = sub_builder2.AddNode("data2", "Data", 0, 0);
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case0);
  sub_graph2->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  case0->GetOpDesc()->AddSubgraphName("branch2");
  case0->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  return root_graph;
}
} // namespace

class UtestGraphUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
*               var                               var
*  atomicclean  |  \                             |   \
*         \\    |   assign                       |   assign
*          \\   |   //         =======>          |   //
*           allreduce                         identity  atomicclean
*             |                                 |       //
*            netoutput                        allreduce
*                                               |
*                                           netoutput
 */
TEST_F(UtestGraphUtils, InsertNodeBefore_DoNotMoveCtrlEdgeFromAtomicClean) {
  // build test graph
  auto builder = ut::GraphBuilder("test");
  const auto &var = builder.AddNode("var", VARIABLE, 0, 1);
  const auto &assign = builder.AddNode("assign", "Assign", 1, 1);
  const auto &allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  const auto &atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  const auto &netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  const auto &identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  // insert identity before allreduce
  GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 0);

  // check assign control-in on identity
  ASSERT_EQ(identity->GetInControlNodes().at(0)->GetName(), "assign");
  // check atomicclean control-in still on allreuce
  ASSERT_EQ(allreduce->GetInControlNodes().at(0)->GetName(), "atomic_clean");
}

TEST_F(UtestGraphUtils, GetSubgraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &case0 = root_builder.AddNode("case0", "Case", 0, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &case1 = sub_builder1.AddNode("case1", "Case", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = ut::GraphBuilder("sub2");
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case1);
  sub_graph2->SetParentGraph(sub_graph1);
  case1->GetOpDesc()->AddSubgraphName("branch1");
  case1->GetOpDesc()->SetSubgraphInstanceName(0, "sub2");
  case1->GetOpDesc()->AddSubgraphName("branch2");
  case1->GetOpDesc()->SetSubgraphInstanceName(1, "not_exist");

  std::vector<ComputeGraphPtr> subgraphs1;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(root_graph, subgraphs1), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs1.size(), 2);

  std::vector<ComputeGraphPtr> subgraphs2;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph1, subgraphs2), GRAPH_SUCCESS);
  ASSERT_EQ(subgraphs2.size(), 1);

  std::vector<ComputeGraphPtr> subgraphs3;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(sub_graph2, subgraphs3), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs3.empty());
}

TEST_F(UtestGraphUtils, GetSubgraphs_nullptr_graph) {
  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_NE(GraphUtils::GetSubgraphsRecursively(nullptr, subgraphs), GRAPH_SUCCESS);
  ASSERT_TRUE(subgraphs.empty());
}

TEST_F(UtestGraphUtils, ReplaceEdgeSrc) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetOutDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetOutControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeSrc(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetOutControlAnchor()), GRAPH_SUCCESS);
}

TEST_F(UtestGraphUtils, ReplaceEdgeDst) {
  auto builder = ut::GraphBuilder("root");
  const auto &node0 = builder.AddNode("node0", "node", 1, 1);
  const auto &node1 = builder.AddNode("node1", "node", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node", 1, 1);
  builder.AddDataEdge(node0, 0, node2, 0);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node1->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0),
                                       node3->GetInDataAnchor(0)), GRAPH_SUCCESS);

  builder.AddControlEdge(node0, node2);
  ASSERT_EQ(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node1->GetInControlAnchor()), GRAPH_SUCCESS);
  ASSERT_NE(GraphUtils::ReplaceEdgeDst(node0->GetOutControlAnchor(), node2->GetInControlAnchor(),
                                       node3->GetInControlAnchor()), GRAPH_SUCCESS);
}

/*
 *          data0  data1
 *             \    /|
 *              add1 | data2
 *                 \ |  /|
 *                  add2 | data3
 *                     \ |  /|
 *                      add3 |  data4
 *                         \ |  / | \
 *                          add4  | cast1
 *                              \ | / |
 *                              add5  |
 *                                | \ |
 *                                | cast2
 *                                | /
 *                             netoutput
 */
TEST_F(UtestGraphUtils, BuildSubgraphWithNodes) {
  auto builder = ut::GraphBuilder("root");
  const auto &data0 = builder.AddNode("data0", DATA, 1, 1);
  const auto &data1 = builder.AddNode("data1", DATA, 1, 1);
  const auto &data2 = builder.AddNode("data2", DATA, 1, 1);
  const auto &data3 = builder.AddNode("data3", DATA, 1, 1);
  const auto &data4 = builder.AddNode("data4", DATA, 1, 1);

  const auto &add1 = builder.AddNode("add1", "Add", 2, 1);
  const auto &add2 = builder.AddNode("add2", "Add", 2, 1);
  const auto &add3 = builder.AddNode("add3", "Add", 2, 1);
  const auto &add4 = builder.AddNode("add4", "Add", 2, 1);
  const auto &add5 = builder.AddNode("add5", "Add", 2, 1);

  const auto &cast1 = builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = builder.AddNode("cast2", "Cast", 1, 1);
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, add1, 0);
  builder.AddDataEdge(data1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(data2, 0, add2, 1);
  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(data3, 0, add3, 1);
  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(data4, 0, add4, 1);
  builder.AddDataEdge(data4, 0, cast1, 0);
  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(cast1, 0, add5, 1);
  builder.AddDataEdge(add5, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, netoutput, 0);

  builder.AddControlEdge(data1, add2);
  builder.AddControlEdge(data2, add3);
  builder.AddControlEdge(data3, add4);
  builder.AddControlEdge(data4, add5);
  builder.AddControlEdge(add5, netoutput);
  builder.AddControlEdge(cast1, cast2);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(nullptr, {}, "subgraph1"), nullptr);

  const auto &graph = builder.GetGraph();
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {}, "subgraph1"), nullptr);

  std::set<NodePtr> nodes = { data1, add2, add3, add4, add5, cast2 };
  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1"), nullptr);

  ASSERT_TRUE(AttrUtils::SetStr(graph, "_session_graph_id", "_session_graph_id"));
  const auto &subgraph1 = GraphUtils::BuildSubgraphWithNodes(graph, nodes, "subgraph1");
  ASSERT_NE(subgraph1, nullptr);
  ASSERT_EQ(subgraph1->GetParentGraph(), graph);
  ASSERT_TRUE(subgraph1->HasAttr("_session_graph_id"));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "data1"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add2"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add3"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add4"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "add5"; }));
  ASSERT_FALSE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetName() == "cast2"; }));
  ASSERT_TRUE(IfNodeExist(graph, [](const NodePtr &node) { return node->GetType() == PARTITIONEDCALL; }));
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);

  ASSERT_EQ(GraphUtils::BuildSubgraphWithNodes(graph, {cast1}, "subgraph1"), nullptr);
}

TEST_F(UtestGraphUtils, UnfoldSubgraph) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForUnfold(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &filter = [](const ComputeGraphPtr &graph) {
    const auto &parent_node = graph->GetParentNode();
    if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
      return false;
    }
    if ((parent_node->GetType() != PARTITIONEDCALL) ||
        (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() != 1)) {
      return false;
    }
    return graph->GetGraphUnknownFlag();
  };
  ASSERT_EQ(GraphUtils::UnfoldSubgraph(subgraph, filter), GRAPH_SUCCESS);

  ASSERT_EQ(graph->GetAllSubgraphs().size(), 1);
  ASSERT_FALSE(graph->GetAllSubgraphs()[0]->GetGraphUnknownFlag());
}

TEST_F(UtestGraphUtils, GetIndependentCompileGraphs) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &partitioned_call0 = root_builder.AddNode("PartitionedCall", "PartitionedCall", 0, 0);
  const auto &root_graph = root_builder.GetGraph();
  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, true);

  auto sub_builder1 = ut::GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("Data", "Data", 0, 0);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(partitioned_call0);
  sub_graph1->SetParentGraph(root_graph);
  partitioned_call0->GetOpDesc()->AddSubgraphName("sub1");
  partitioned_call0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "sub1");

  (void)AttrUtils::SetBool(*root_graph, ATTR_NAME_PIPELINE_PARTITIONED, false);
  independent_compile_subgraphs.clear();
  ASSERT_EQ(GraphUtils::GetIndependentCompileGraphs(root_graph, independent_compile_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(independent_compile_subgraphs.size(), 1);
  ASSERT_EQ(independent_compile_subgraphs[0]->GetName(), "root");
}

TEST_F(UtestGraphUtils, InsertNodeAfter) {
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();

  auto graph_builder1 = ut::GraphBuilder("test_graph1");
  const auto &node1 = graph_builder1.AddNode("data1", DATA, 1, 1);
  const auto &graph1 = graph_builder1.GetGraph();

  std::vector<ComputeGraphPtr> independent_compile_subgraphs;
  ASSERT_EQ(GraphUtils::InsertNodeAfter(node0->GetOutDataAnchor(0), {}, node1, 0, 0), GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, CheckDumpGraphNum) {
  std::map<std::string, std::string> session_option{{"ge.maxDumpFileNum", "2"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  auto graph_builder0 = ut::GraphBuilder("test_graph0");
  const auto &node0 = graph_builder0.AddNode("data0", DATA, 1, 1);
  const auto &graph0 = graph_builder0.GetGraph();
  GraphUtils::DumpGEGrph(graph0, "./", "1");
  GraphUtils::DumpGEGrph(graph0, "./", "1");
  GraphUtils::DumpGEGrph(graph0, "./", "1");
}

TEST_F(UtestGraphUtils, CopyRootComputeGraph) {
  auto graph = BuildGraphWithSubGraph();
  // check origin graph size
  ASSERT_EQ(graph->GetAllNodesSize(), 3);
  ComputeGraphPtr dst_compute_graph = std::make_shared<ComputeGraph>(ComputeGraph("dst"));
  // test copy root graph success
  auto ret = GraphUtils::CopyComputeGraph(graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ASSERT_EQ(dst_compute_graph->GetAllNodesSize(), 3);
  // test copy subgraph failed
  auto sub1_graph = graph->GetSubgraph("sub1");
  ret = GraphUtils::CopyComputeGraph(sub1_graph, dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);

  // test copy dst compute_graph null
  ComputeGraphPtr empty_dst_compute_graph;
  ret = GraphUtils::CopyComputeGraph(graph, empty_dst_compute_graph);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphUtils, DumpGraphByPath) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 0);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  // test dump_level 0
  auto ret = GraphUtils::DumpGEGraphByPath(graph, "./test/test_graph_0.txt", 0);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "/", 0);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "test_graph_0.txt", 0);
  ASSERT_EQ((ret != 0), true);
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_0.txt", 0);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph0 = std::make_shared<ComputeGraph>("TestGraph0");
  bool state = GraphUtils::LoadGEGraph("./test_graph_0.txt", *com_graph0);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph0->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph0->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), false);
    }
  }

  // test dump_level 1
  ret = GraphUtils::DumpGEGraphByPath(graph, "./test_graph_1.txt", 1);
  ASSERT_EQ(ret, 0);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  state = GraphUtils::LoadGEGraph("./test_graph_1.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 4);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), true);
      ASSERT_EQ((ge_tensor_ptr == nullptr), false);
      const TensorData tensor_data = ge_tensor_ptr->GetData();
      const uint8_t *buff = tensor_data.GetData();
      ASSERT_EQ((buff == nullptr), false);
      ASSERT_EQ(buff[0], 7);
      ASSERT_EQ(buff[10], 8);
    }
  }
}

}  // namespace ge
