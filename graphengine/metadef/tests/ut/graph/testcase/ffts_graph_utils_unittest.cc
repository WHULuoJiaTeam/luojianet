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

#include "graph/utils/ffts_graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/ge_inner_error_codes.h"

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

NodePtr FindNodeWithNamePattern(const ComputeGraphPtr &graph, const std::string &pattern,
                                bool direct_node_flag = true) {
  for (const auto &node : graph->GetNodes(direct_node_flag)) {
    const auto &name = node->GetName();
    if (name.find(pattern) != string::npos) {
      return node;
    }
  }
  return nullptr;
}

void GetSubgraphsWithFilter(const ComputeGraphPtr &graph, std::function<bool(const ComputeGraphPtr &)> filter,
                            std::vector<ComputeGraphPtr> &subgraphs) {
  for (const auto &subgraph : graph->GetAllSubgraphs()) {
    if (filter(subgraph)) {
      subgraphs.emplace_back(subgraph);
    }
  }
}

bool IsAllNodeMatch(const ComputeGraphPtr &graph, std::function<bool(const NodePtr &)> filter,
                    bool direct_node_flag = true) {
  for (const auto &node : graph->GetNodes(direct_node_flag)) {
    if (!filter(node)) {
      return false;
    }
  }
  return true;
}

/*
 *                    data
 *                      |
 *                    cast1
 *                      |
 *                    cast2
 *                      |
 *                    cast3
 *                      |
 *                    cast4
 *                      |
 *                    cast5
 *                      |
 *                    cast6
 *                      |
 *                  netoutput
 */
void BuildGraphForSplit_without_func_node(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto sub_builder = ut::GraphBuilder("subgraph");
  const auto &data1 = sub_builder.AddNode("data1", DATA, 1, 1);
  const auto &cast1 = sub_builder.AddNode("cast1", "Cast", 1, 1);
  const auto &cast2 = sub_builder.AddNode("cast2", "Cast", 1, 1);
  const auto &cast3 = sub_builder.AddNode("cast3", "Cast", 1, 1);
  const auto &cast4 = sub_builder.AddNode("cast4", "Cast", 1, 1);
  const auto &cast5 = sub_builder.AddNode("cast5", "Cast", 1, 1);
  const auto &cast6 = sub_builder.AddNode("cast6", "Cast", 1, 1);
  const auto &netoutput = sub_builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  sub_builder.AddDataEdge(data1, 0, cast1, 0);
  sub_builder.AddDataEdge(cast1, 0, cast2, 0);
  sub_builder.AddDataEdge(cast2, 0, cast3, 0);
  sub_builder.AddDataEdge(cast3, 0, cast4, 0);
  sub_builder.AddDataEdge(cast4, 0, cast5, 0);
  sub_builder.AddDataEdge(cast5, 0, cast6, 0);
  sub_builder.AddDataEdge(cast6, 0, netoutput, 0);

  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);

  auto builder = ut::GraphBuilder("root");
  const auto &input = builder.AddNode("data", DATA, 1, 1);
  const auto &func_node = builder.AddNode("func_node", PARTITIONEDCALL, 1, 1);
  const auto &output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  subgraph = sub_builder.GetGraph();
  AttrUtils::SetStr(subgraph, "_session_graph_id", "_session_graph_id");
  graph = builder.GetGraph();
  AttrUtils::SetStr(graph, "_session_graph_id", "_session_graph_id");

  func_node->GetOpDesc()->AddSubgraphName("f");
  func_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  AttrUtils::SetStr(func_node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, "ffts_plus");
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func_node);
  subgraph->SetParentGraph(graph);

  return;
}

/*
 * ********** root ********** func ********** then_1 ********** else_1 ********** then_2 ********** else_2 **********
 *
 *            input        var1  data0        data1             data2             data3             data4
 *              |             \   /             |                 |                 |                 |
 *            func   constant  if1            cast1             cast2             cast3             cast4
 *              |          \   /                |                 |                 |                 |
 *           output        less              square1           square2           square3           square4
 *                           |                  |                 |                 |                 |
 *                       netoutput0         netoutput1    var2   if2            netoutput3        netoutput4
 *                                                            \ /
 *                                                         netoutput2
 *
 * ******************************************************************************************************************
 */
void BuildGraphForSplit_with_func_node(ComputeGraphPtr &graph, ComputeGraphPtr &subgraph) {
  auto builder = ut::GraphBuilder("root");
  const auto &input = builder.AddNode("input", DATA, 1, 1);
  const auto &func = builder.AddNode("func", PARTITIONEDCALL, 1, 1);
  const auto &output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(input, 0, func, 0);
  builder.AddDataEdge(func, 0, output, 0);
  graph = builder.GetGraph();
  AttrUtils::SetStr(graph, "_session_graph_id", "_session_graph_id");

  auto sub_builder0 = ut::GraphBuilder("func");
  const auto &data0 = sub_builder0.AddNode("data0", DATA, 1, 1);
  const auto &var1 = sub_builder0.AddNode("var1", VARIABLEV2, 1, 1);
  const auto &if1 = sub_builder0.AddNode("if1", "If", 2, 1);
  const auto &constant = sub_builder0.AddNode("constant", CONSTANTOP, 1, 1);
  const auto &less = sub_builder0.AddNode("less", "Less", 2, 1);
  const auto &netoutput0 = sub_builder0.AddNode("netoutput0", NETOUTPUT, 1, 0);
  sub_builder0.AddDataEdge(var1, 0, if1, 0);
  sub_builder0.AddDataEdge(data0, 0, if1, 1);
  sub_builder0.AddDataEdge(constant, 0, less, 0);
  sub_builder0.AddDataEdge(if1, 0, less, 1);
  sub_builder0.AddDataEdge(less, 0, netoutput0, 0);
  AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput0->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  subgraph = sub_builder0.GetGraph();
  AttrUtils::SetStr(subgraph, "_session_graph_id", "_session_graph_id");
  func->GetOpDesc()->AddSubgraphName("f");
  func->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  AttrUtils::SetStr(func->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, "ffts_plus");
  graph->AddSubGraph(subgraph);
  subgraph->SetParentNode(func);
  subgraph->SetParentGraph(graph);

  auto sub_builder1 = ut::GraphBuilder("then_1");
  const auto &data1 = sub_builder1.AddNode("data1", DATA, 1, 1);
  const auto &cast1 = sub_builder1.AddNode("cast1", "Cast", 1, 1);
  const auto &square1 = sub_builder1.AddNode("square1", "Square", 1, 1);
  const auto &netoutput1 = sub_builder1.AddNode("netoutput1", NETOUTPUT, 1, 0);
  sub_builder1.AddDataEdge(data1, 0, cast1, 0);
  sub_builder1.AddDataEdge(cast1, 0, square1, 0);
  sub_builder1.AddDataEdge(square1, 0, netoutput1, 0);
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput1->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto &subgraph1 = sub_builder1.GetGraph();
  AttrUtils::SetStr(subgraph1, "_session_graph_id", "_session_graph_id");
  if1->GetOpDesc()->AddSubgraphName("then_branch");
  if1->GetOpDesc()->SetSubgraphInstanceName(0, subgraph1->GetName());
  graph->AddSubGraph(subgraph1);
  subgraph1->SetParentNode(if1);
  subgraph1->SetParentGraph(subgraph);

  auto sub_builder2 = ut::GraphBuilder("else_1");
  const auto &data2 = sub_builder2.AddNode("data2", DATA, 1, 1);
  const auto &cast2 = sub_builder2.AddNode("cast2", "Cast", 1, 1);
  const auto &square2 = sub_builder2.AddNode("square2", "Square", 1, 1);
  const auto &var2 = sub_builder2.AddNode("var2", VARIABLEV2, 1, 1);
  const auto &if2 = sub_builder2.AddNode("if2", "If", 2, 1);
  const auto &netoutput2 = sub_builder2.AddNode("netoutput2", NETOUTPUT, 1, 0);
  sub_builder2.AddDataEdge(data2, 0, cast2, 0);
  sub_builder2.AddDataEdge(cast2, 0, square2, 0);
  sub_builder2.AddDataEdge(square2, 0, if2, 1);
  sub_builder2.AddDataEdge(var2, 0, if2, 0);
  sub_builder2.AddDataEdge(if2, 0, netoutput2, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto &subgraph2 = sub_builder2.GetGraph();
  AttrUtils::SetStr(subgraph2, "_session_graph_id", "_session_graph_id");
  if1->GetOpDesc()->AddSubgraphName("else_branch");
  if1->GetOpDesc()->SetSubgraphInstanceName(1, subgraph2->GetName());
  graph->AddSubGraph(subgraph2);
  subgraph2->SetParentNode(if1);
  subgraph2->SetParentGraph(subgraph);

  auto sub_builder3 = ut::GraphBuilder("then_2");
  const auto &data3 = sub_builder3.AddNode("data3", DATA, 1, 1);
  const auto &cast3 = sub_builder3.AddNode("cast3", "Cast", 1, 1);
  const auto &square3 = sub_builder3.AddNode("square3", "Square", 1, 1);
  const auto &netoutput3 = sub_builder3.AddNode("netoutput3", NETOUTPUT, 1, 0);
  sub_builder3.AddDataEdge(data3, 0, cast3, 0);
  sub_builder3.AddDataEdge(cast3, 0, square3, 0);
  sub_builder1.AddDataEdge(square3, 0, netoutput3, 0);
  AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput3->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto &subgraph3 = sub_builder3.GetGraph();(AttrUtils::SetStr(subgraph3, "_session_graph_id", "_session_graph_id"));
  if2->GetOpDesc()->AddSubgraphName("then_branch");
  if2->GetOpDesc()->SetSubgraphInstanceName(0, subgraph3->GetName());
  graph->AddSubGraph(subgraph3);
  subgraph3->SetParentNode(if2);
  subgraph3->SetParentGraph(subgraph2);

  auto sub_builder4 = ut::GraphBuilder("else_2");
  const auto &data4 = sub_builder4.AddNode("data4", DATA, 1, 1);
  const auto &cast4 = sub_builder4.AddNode("cast4", "Cast", 1, 1);
  const auto &square4 = sub_builder4.AddNode("square4", "Square", 1, 1);
  const auto &netoutput4 = sub_builder4.AddNode("netoutput4", NETOUTPUT, 1, 0);
  sub_builder4.AddDataEdge(data4, 0, cast4, 0);
  sub_builder4.AddDataEdge(cast4, 0, square4, 0);
  sub_builder4.AddDataEdge(square4, 0, netoutput4, 0);
  AttrUtils::SetInt(data4->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(netoutput4->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto &subgraph4 = sub_builder4.GetGraph();
  AttrUtils::SetStr(subgraph4, "_session_graph_id", "_session_graph_id");
  if2->GetOpDesc()->AddSubgraphName("else_branch");
  if2->GetOpDesc()->SetSubgraphInstanceName(1, subgraph4->GetName());
  graph->AddSubGraph(subgraph4);
  subgraph4->SetParentNode(if2);
  subgraph4->SetParentGraph(subgraph2);

  return;
}
}

class UtestFftsGraphUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFftsGraphUtils, LimitExceedPartition_invalid_input) {
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*ut::GraphBuilder("root").GetGraph(), nullptr, {}), GRAPH_SUCCESS);
  const auto &calc_func = [](const NodePtr &node) {
    return std::vector<uint32_t> {1};
  };
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*ut::GraphBuilder("root").GetGraph(), calc_func, {}), GRAPH_SUCCESS);
}

TEST_F(UtestFftsGraphUtils, LimitExceedPartition_no_func_node) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForSplit_without_func_node(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &calc_func = [](const NodePtr &node) {
    return std::vector<uint32_t> {1};
  };
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, calc_func, {8}), GRAPH_SUCCESS);
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, calc_func, {3}), GRAPH_SUCCESS);
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  ASSERT_EQ(graph->GetAllSubgraphs().size(), 2);
  std::vector<ComputeGraphPtr> subgraphs;
  GetSubgraphsWithFilter(graph,
                         [](const ComputeGraphPtr &graph) {
                           const auto &parent_node = graph->GetParentNode();
                           if ((parent_node == nullptr) || (parent_node->GetOpDesc() == nullptr)) {
                             return false;
                           }
                           return parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH); },
                         subgraphs);
  ASSERT_EQ(subgraphs.size(), 2);
  for (const auto &subgraph : subgraphs) {
    ASSERT_TRUE(subgraph != nullptr);
    ASSERT_TRUE(IsAllNodeMatch(subgraph,
                               [](const NodePtr &node) {
                                 return node->GetOpDesc()->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
                               }, false));
  }

  const auto &cast1 = FindNodeWithNamePattern(graph, "cast1", false);
  ASSERT_NE(cast1, nullptr);
  const auto &cast2 = FindNodeWithNamePattern(graph, "cast2", false);
  ASSERT_NE(cast2, nullptr);
  const auto &cast3 = FindNodeWithNamePattern(graph, "cast3", false);
  ASSERT_NE(cast3, nullptr);
  const auto &cast4 = FindNodeWithNamePattern(graph, "cast4", false);
  ASSERT_NE(cast4, nullptr);
  const auto &cast5 = FindNodeWithNamePattern(graph, "cast5", false);
  ASSERT_NE(cast5, nullptr);
  const auto &cast6 = FindNodeWithNamePattern(graph, "cast6", false);
  ASSERT_NE(cast6, nullptr);
  const auto &func1 = subgraphs[0]->GetParentNode();
  ASSERT_NE(func1, nullptr);
  const auto &func2 = subgraphs[1]->GetParentNode();
  ASSERT_NE(func2, nullptr);
  ASSERT_EQ(cast1->GetOwnerComputeGraph(), cast2->GetOwnerComputeGraph());
  ASSERT_EQ(cast1->GetOwnerComputeGraph(), cast3->GetOwnerComputeGraph());
  ASSERT_EQ(cast4->GetOwnerComputeGraph(), cast5->GetOwnerComputeGraph());
  ASSERT_EQ(cast4->GetOwnerComputeGraph(), cast6->GetOwnerComputeGraph());
  ASSERT_NE(cast1->GetOwnerComputeGraph(), cast4->GetOwnerComputeGraph());
  ASSERT_NE(cast1->GetOwnerComputeGraph(), graph);
  ASSERT_NE(cast4->GetOwnerComputeGraph(), graph);
  ASSERT_EQ(func1->GetOwnerComputeGraph(), graph);
  ASSERT_EQ(func2->GetOwnerComputeGraph(), graph);

  const auto &input = graph->FindFirstNodeMatchType(DATA);
  ASSERT_NE(input, nullptr);
  const auto &output = graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(input->GetOutDataAnchor(0)->IsLinkedWith(func1->GetInDataAnchor(0)));
  ASSERT_TRUE(func1->GetOutDataAnchor(0)->IsLinkedWith(func2->GetInDataAnchor(0)));
  ASSERT_TRUE(func2->GetOutDataAnchor(0)->IsLinkedWith(output->GetInDataAnchor(0)));
  const auto &data1 = subgraphs[0]->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data1, nullptr);
  const auto &netoutput1 = subgraphs[0]->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(netoutput1, nullptr);
  ASSERT_TRUE(data1->GetOutDataAnchor(0)->IsLinkedWith(cast1->GetInDataAnchor(0)));
  ASSERT_TRUE(cast1->GetOutDataAnchor(0)->IsLinkedWith(cast2->GetInDataAnchor(0)));
  ASSERT_TRUE(cast2->GetOutDataAnchor(0)->IsLinkedWith(cast3->GetInDataAnchor(0)));
  ASSERT_TRUE(cast3->GetOutDataAnchor(0)->IsLinkedWith(netoutput1->GetInDataAnchor(0)));
  const auto &data2 = subgraphs[1]->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data2, nullptr);
  const auto &netoutput2 = subgraphs[1]->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(netoutput2, nullptr);
  ASSERT_TRUE(data2->GetOutDataAnchor(0)->IsLinkedWith(cast4->GetInDataAnchor(0)));
  ASSERT_TRUE(cast4->GetOutDataAnchor(0)->IsLinkedWith(cast5->GetInDataAnchor(0)));
  ASSERT_TRUE(cast5->GetOutDataAnchor(0)->IsLinkedWith(cast6->GetInDataAnchor(0)));
  ASSERT_TRUE(cast6->GetOutDataAnchor(0)->IsLinkedWith(netoutput2->GetInDataAnchor(0)));
}

TEST_F(UtestFftsGraphUtils, LimitExceedPartition_with_func_node) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForSplit_with_func_node(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  const auto &calc_func = [](const NodePtr &node) {
    return std::vector<uint32_t> {1};
  };
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, calc_func, {8}), GRAPH_SUCCESS);
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
  ASSERT_EQ(graph->GetAllSubgraphs().size(), 9);
}

//TEST_F(UtestFftsGraphUtils, ClipNodesFromGraph_no_func_node) {
//  ComputeGraphPtr graph;
//  ComputeGraphPtr subgraph;
//  BuildGraphForSplit_without_func_node(graph, subgraph);
//  ASSERT_NE(graph, nullptr);
//  ASSERT_NE(subgraph, nullptr);
//
//  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, {}), GRAPH_SUCCESS);
//
//  auto data1 = FindNodeWithNamePattern(subgraph, "data1");
//  ASSERT_NE(data1.get(), nullptr);
//  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, {data1}), GRAPH_SUCCESS);
//
//  std::set<NodePtr> unsupported_nodes;
//  auto cast1 = FindNodeWithNamePattern(subgraph, "cast1");
//  ASSERT_NE(cast1, nullptr);
//  unsupported_nodes.insert(cast1);
//  auto cast4 = FindNodeWithNamePattern(subgraph, "cast4");
//  ASSERT_NE(cast4, nullptr);
//  unsupported_nodes.insert(cast4);
//  auto cast5 = FindNodeWithNamePattern(subgraph, "cast5");
//  ASSERT_NE(cast5, nullptr);
//  unsupported_nodes.insert(cast5);
//  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, unsupported_nodes), GRAPH_SUCCESS);
//  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);
//
//  ASSERT_EQ(graph->GetAllSubgraphs().size(), 3);
//  const auto &parent_node = subgraph->GetParentNode();
//  ASSERT_NE(parent_node, nullptr);
//  ASSERT_FALSE(parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH));
//  ASSERT_TRUE(IsAllNodeMatch(subgraph,
//                             [](const NodePtr &node) {
//                               return !node->GetOpDesc()->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
//                             }));
//
//  std::vector<ComputeGraphPtr> subgraphs;
//  GetSubgraphsWithFilter(graph,
//                         [](const ComputeGraphPtr &graph) {
//                           const auto &parent_node = graph->GetParentNode();
//                           if ((parent_node == nullptr) || (parent_node->GetOpDesc() == nullptr)) {
//                             return false;
//                           }
//                           return parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH); },
//                         subgraphs);
//  ASSERT_EQ(subgraphs.size(), 2);
//  for (const auto &subgraph : subgraphs) {
//    ASSERT_TRUE(subgraph != nullptr);
//    ASSERT_TRUE(IsAllNodeMatch(subgraph,
//                               [](const NodePtr &node) {
//                                 return node->GetOpDesc()->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
//                               }, false));
//  }
//
//  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "data1"; }));
//  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast1"; }));
//  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast4"; }));
//  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast5"; }));
//  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "netoutput"; }));
//  ASSERT_FALSE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast2"; }));
//  ASSERT_FALSE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast3"; }));
//  ASSERT_FALSE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "cast6"; }));
//
//  const auto &cast2 = FindNodeWithNamePattern(graph, "cast2", false);
//  ASSERT_NE(cast2, nullptr);
//  const auto &cast3 = FindNodeWithNamePattern(graph, "cast3", false);
//  ASSERT_NE(cast3, nullptr);
//  const auto &cast6 = FindNodeWithNamePattern(graph, "cast6", false);
//  ASSERT_NE(cast6, nullptr);
//  ASSERT_EQ(cast2->GetOwnerComputeGraph(), cast3->GetOwnerComputeGraph());
//  ASSERT_NE(cast2->GetOwnerComputeGraph(), cast6->GetOwnerComputeGraph());
//}

TEST_F(UtestFftsGraphUtils, ClipNodesFromGraph_with_func_node) {
  ComputeGraphPtr graph;
  ComputeGraphPtr subgraph;
  BuildGraphForSplit_with_func_node(graph, subgraph);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(subgraph, nullptr);

  std::set<NodePtr> unsupported_nodes;
  const auto &cast1 = FindNodeWithNamePattern(graph, "cast1", false);
  ASSERT_NE(cast1, nullptr);
  unsupported_nodes.insert(cast1);
  const auto &cast3 = FindNodeWithNamePattern(graph, "cast3", false);
  ASSERT_NE(cast3, nullptr);
  unsupported_nodes.insert(cast3);
  ASSERT_EQ(FftsGraphUtils::GraphPartition(*subgraph, unsupported_nodes), GRAPH_SUCCESS);
  ASSERT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  ASSERT_EQ(graph->GetAllSubgraphs().size(), 10);
  const auto &parent_node = subgraph->GetParentNode();
  ASSERT_NE(parent_node, nullptr);
  ASSERT_FALSE(parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH));
  ASSERT_TRUE(IsAllNodeMatch(subgraph,
                             [](const NodePtr &node) {
                               return !node->GetOpDesc()->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
                             }));

  std::vector<ComputeGraphPtr> subgraphs;
  GetSubgraphsWithFilter(graph,
                         [](const ComputeGraphPtr &graph) {
                           const auto &parent_node = graph->GetParentNode();
                           if ((parent_node == nullptr) || (parent_node->GetOpDesc() == nullptr)) {
                             return false;
                           }
                           return parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH); },
                         subgraphs);
  ASSERT_EQ(subgraphs.size(), 5);
  for (const auto &subgraph : subgraphs) {
    ASSERT_TRUE(subgraph != nullptr);
    ASSERT_TRUE(IsAllNodeMatch(subgraph,
                               [](const NodePtr &node) {
                                 return node->GetOpDesc()->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
                               }, false));
  }

  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "var1"; }));
  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "data0"; }));
  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "constant"; }));
  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "if1"; }));
  ASSERT_FALSE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "less"; }));
  ASSERT_TRUE(IfNodeExist(subgraph, [](const NodePtr &node) { return node->GetName() == "netoutput0"; }));

  const auto &if1 = FindNodeWithNamePattern(subgraph, "if1");
  ASSERT_NE(if1, nullptr);
  std::vector<ComputeGraphPtr> if1_subgraphs;
  ASSERT_EQ(NodeUtils::GetDirectSubgraphs(if1, if1_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(if1_subgraphs.size(), 2);
  const auto &then1 = if1_subgraphs[0];
  ASSERT_NE(then1, nullptr);
  ASSERT_TRUE(IfNodeExist(then1, [](const NodePtr &node) { return node->GetName() == "data1"; }));
  ASSERT_TRUE(IfNodeExist(then1, [](const NodePtr &node) { return node->GetName() == "cast1"; }));
  ASSERT_FALSE(IfNodeExist(then1, [](const NodePtr &node) { return node->GetName() == "square1"; }));
  ASSERT_TRUE(IfNodeExist(then1, [](const NodePtr &node) { return node->GetName() == "netoutput1"; }));
  const auto &else1 = if1_subgraphs[1];
  ASSERT_NE(else1, nullptr);
  ASSERT_TRUE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "data2"; }));
  ASSERT_TRUE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "var2"; }));
  ASSERT_FALSE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "cast2"; }));
  ASSERT_FALSE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "square2"; }));
  ASSERT_TRUE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "if2"; }));
  ASSERT_TRUE(IfNodeExist(else1, [](const NodePtr &node) { return node->GetName() == "netoutput2"; }));

  const auto &if2 = FindNodeWithNamePattern(else1, "if2");
  ASSERT_NE(if2, nullptr);
  std::vector<ComputeGraphPtr> if2_subgraphs;
  ASSERT_EQ(NodeUtils::GetDirectSubgraphs(if2, if2_subgraphs), GRAPH_SUCCESS);
  ASSERT_EQ(if2_subgraphs.size(), 2);
  const auto &then2 = if2_subgraphs[0];
  ASSERT_NE(then2, nullptr);
  ASSERT_TRUE(IfNodeExist(then2, [](const NodePtr &node) { return node->GetName() == "data3"; }));
  ASSERT_TRUE(IfNodeExist(then2, [](const NodePtr &node) { return node->GetName() == "cast3"; }));
  ASSERT_FALSE(IfNodeExist(then2, [](const NodePtr &node) { return node->GetName() == "square3"; }));
  ASSERT_TRUE(IfNodeExist(then2, [](const NodePtr &node) { return node->GetName() == "netoutput3"; }));
  const auto &else2 = if2_subgraphs[1];
  ASSERT_NE(else2, nullptr);
  ASSERT_TRUE(IfNodeExist(else2, [](const NodePtr &node) { return node->GetName() == "data4"; }));
  ASSERT_FALSE(IfNodeExist(else2, [](const NodePtr &node) { return node->GetName() == "cast4"; }));
  ASSERT_FALSE(IfNodeExist(else2, [](const NodePtr &node) { return node->GetName() == "square4"; }));
  ASSERT_TRUE(IfNodeExist(else2, [](const NodePtr &node) { return node->GetName() == "netoutput4"; }));
}

TEST_F(UtestFftsGraphUtils, CheckRecursionDepth) {
  std::map<NodePtr, std::vector<uint32_t>> node_value;
  std::map<ComputeGraphPtr , std::vector<uint32_t>> graph_value;
  ComputeGraphPtr graph = nullptr;
  ASSERT_EQ(FftsGraphUtils::Calculate(graph, nullptr, node_value, graph_value, 10), GRAPH_FAILED);
  ASSERT_EQ(FftsGraphUtils::Calculate(graph, nullptr, node_value, graph_value, 9), PARAM_INVALID);
  ASSERT_EQ(FftsGraphUtils::PartitionGraphWithLimit(nullptr, node_value, graph_value, {}, 10), GRAPH_FAILED);
  ASSERT_EQ(FftsGraphUtils::PartitionGraphWithLimit(nullptr, node_value, graph_value, {}, 9), PARAM_INVALID);
}

TEST_F(UtestFftsGraphUtils, SplitSubgraph_nullptr_graph) {
  std::vector<std::pair<bool, std::set<NodePtr>>> split_nodes;
  split_nodes.emplace_back(std::make_pair(true, std::set<NodePtr>{ nullptr }));
  ASSERT_EQ(FftsGraphUtils::SplitSubgraph(nullptr, split_nodes), GRAPH_FAILED);
}

TEST_F(UtestFftsGraphUtils, SetAttrForFftsPlusSubgraph_nullptr_parent_node) {
  auto builder = ut::GraphBuilder("");
  ASSERT_EQ(FftsGraphUtils::SetAttrForFftsPlusSubgraph(builder.GetGraph()), GRAPH_FAILED);
}

TEST_F(UtestFftsGraphUtils, Calculate_nullptr_node) {
  NodePtr node = nullptr;
  std::map<NodePtr, std::vector<uint32_t>> node_value;
  std::map<ComputeGraphPtr , std::vector<uint32_t>> graph_value;
  ASSERT_TRUE(FftsGraphUtils::Calculate(node, nullptr, node_value, graph_value, 1).empty());
}
}  // namespace ge
