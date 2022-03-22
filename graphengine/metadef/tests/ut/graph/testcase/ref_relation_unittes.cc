#include "graph/ref_relation.h"

#include <gtest/gtest.h>
#include <iostream>
#define protected public
#define private public
#include "graph_builder_utils.h"
#include "graph/node.h"
#include "graph/operator_factory.h"
#include "graph/compute_graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#undef protected
#undef private

using namespace ge;
using namespace std;
namespace ge {
class UTTEST_RefRelations : public testing::Test {
 protected:

  void SetUp() {
  }

  void TearDown() {
  }
};

namespace {

/*
 *   netoutput1
 *       |
 *      add
 *     /   \
 * data1   data2
 */
ComputeGraphPtr BuildSubGraph(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto add = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 1, 1);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput, 0);

  return builder.GetGraph();
}
/*
 *   netoutput
 *       |
 *      if
 *     /   \
 * data1   data2
 */
ComputeGraphPtr BuildMainGraphWithIf() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto if1 = builder.AddNode("if", "If", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(if1, 0, netoutput1, 0);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildSubGraph("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}

/*
 *   netoutput
 *       |
 *      if
 *     /   \
 * data1   data2
 */
ComputeGraphPtr BuildMainGraphWithIfButWithNoSubgraph() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto if1 = builder.AddNode("if", "If", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(if1, 0, netoutput1, 0);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub2 = BuildSubGraph("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");

  return main_graph;
}

/*
 *   netoutput1
 *       |
 *      add
 *     /   \       \
 * data1   data2   data3
 */
ComputeGraphPtr BuildSubGraph3(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto data3 = builder.AddNode(name + "data3", "Data", 1, 1);

  auto add = builder.AddNode(name + "sub", "Sub", 3, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 1, 1);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(data3->GetOpDesc(), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(data3, 0, add, 2);
  builder.AddDataEdge(add, 0, netoutput, 0);

  return builder.GetGraph();
}
/*
 *   netoutput
 *       |
 *      if
 *     /   \     \
 * data1   data2  data3
 */
ComputeGraphPtr BuildMainGraphWithIf3() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto data3 = builder.AddNode("data3", "Data", 1, 1);
  auto if1 = builder.AddNode("if", "If", 3, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(data3, 0, if1, 2);
  builder.AddDataEdge(if1, 0, netoutput1, 0);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph3("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(if1);
  if1->GetOpDesc()->AddSubgraphName("sub1");
  if1->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildSubGraph3("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(if1);
  if1->GetOpDesc()->AddSubgraphName("sub2");
  if1->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}

/*
 *   netoutput1
 *       |      \
 *      sub      relu
 *     /   \     /
 * data1   data2
 */
ComputeGraphPtr BuildSubGraph2(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto sub = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto relu = builder.AddNode(name + "relu", "Relu", 1, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 2, 2);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(1));


  builder.AddDataEdge(data1, 0, sub, 0);
  builder.AddDataEdge(data2, 0, sub, 1);
  builder.AddDataEdge(sub, 0, netoutput, 0);
  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 1);


  return builder.GetGraph();
}
/*
 *   netoutput relu
 *       |    /
 *      if
 *     /   \
 * data1   data2
 */
ComputeGraphPtr BuildMainGraphWithIf2() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto if1 = builder.AddNode("if", "If", 2, 2);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 2);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(if1, 0, netoutput1, 0);
  builder.AddDataEdge(if1, 1, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph2("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildSubGraph2("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}
/*
 *   netoutput
 *       |      \
 *      sub      relu   \
 *     /   \     /
 * data1   data2        data3
 */
ComputeGraphPtr BuildWhileBodySubGraph(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto data3 = builder.AddNode(name + "data3", "Data", 1, 1);
  auto sub = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto relu = builder.AddNode(name + "relu", "Relu", 1, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 3, 3);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(data3->GetOpDesc(), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(2), "_parent_node_index", static_cast<int>(2));


  builder.AddDataEdge(data1, 0, sub, 0);
  builder.AddDataEdge(data2, 0, sub, 1);
  builder.AddDataEdge(sub, 0, netoutput, 0);
  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 1);
  builder.AddDataEdge(data3, 0, netoutput, 2);


  return builder.GetGraph();
}
/*
 *   netoutput1
 *       |
 *      mul
 *     /   \      \
 * data1   data2  data3
 */
ComputeGraphPtr BuildWhileCondSubGraph(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto data3 = builder.AddNode(name + "data3", "Data", 1, 1);
  auto mul = builder.AddNode(name + "mul", "Mul", 3, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 1, 1);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(data3->GetOpDesc(), "_parent_node_index", static_cast<int>(2));

  builder.AddDataEdge(data1, 0, mul, 0);
  builder.AddDataEdge(data2, 0, mul, 1);
  builder.AddDataEdge(data3, 0, mul, 2);
  builder.AddDataEdge(mul, 0, netoutput, 0);

  return builder.GetGraph();
}
/*
 *   netoutput relu
 *       |    /
 *      while
 *     /     \     \
 * data1   data2   const
 */
ComputeGraphPtr BuildMainGraphWithWhile() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto n = builder.AddNode("n", "Const", 1, 1);
  auto while1 = builder.AddNode("while1", "While", 3, 3);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 2);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);

  builder.AddDataEdge(data1, 0, while1, 0);
  builder.AddDataEdge(data2, 0, while1, 1);
  builder.AddDataEdge(n, 0, while1, 2);
  builder.AddDataEdge(while1, 0, netoutput1, 0);
  builder.AddDataEdge(while1, 1, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildWhileCondSubGraph("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildWhileBodySubGraph("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}
/*
 *   netoutput
 *       |      \
 *      sub      relu   \
 *     /   \     /
 * data1   data2        data3
 */
ComputeGraphPtr BuildWhileBodySubGraph2(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto data3 = builder.AddNode(name + "data3", "Data", 1, 1);
  auto sub = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto relu = builder.AddNode(name + "relu", "Relu", 1, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 3, 3);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(data3->GetOpDesc(), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(2), "_parent_node_index", static_cast<int>(2));


  builder.AddDataEdge(data1, 0, sub, 0);
  builder.AddDataEdge(data2, 0, sub, 1);
  builder.AddDataEdge(sub, 0, netoutput, 0);
  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 1);
  builder.AddDataEdge(data3, 0, netoutput, 2);


  return builder.GetGraph();
}
/*
 *   netoutput relu
 *       |    /
 *      while
 *     /     \     \
 * data1   data2   const
 */
ComputeGraphPtr BuildMainGraphWithWhile2() {
  ut::GraphBuilder builder("main_graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto n = builder.AddNode("n", "Const", 1, 1);
  auto while1 = builder.AddNode("while1", "While", 3, 3);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 2);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);

  builder.AddDataEdge(data1, 0, while1, 0);
  builder.AddDataEdge(data2, 0, while1, 1);
  builder.AddDataEdge(n, 0, while1, 2);
  builder.AddDataEdge(while1, 0, netoutput1, 0);
  builder.AddDataEdge(while1, 1, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildWhileCondSubGraph("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildWhileBodySubGraph2("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}

/*
 *   netoutput
 *       |      \     \
 *      sub      \     \
 *     /   \      \     \
 * data1   const data2  data3
 */
ComputeGraphPtr BuildWhileBodySubGraph3(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data0 = builder.AddNode(name + "data0", "Data", 1, 1);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto sub = builder.AddNode(name + "sub", "Sub", 2, 1);
  auto const1 = builder.AddNode(name + "const1", "Const", 0, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 4, 4);

  AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(2), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));


  builder.AddDataEdge(data0, 0, sub, 0);
  builder.AddDataEdge(const1, 0, sub, 1);
  builder.AddDataEdge(const1, 0, netoutput, 3);
  builder.AddDataEdge(sub, 0, netoutput, 2);
  builder.AddDataEdge(data1, 0, netoutput, 1);
  builder.AddDataEdge(data2, 0, netoutput, 0);


  return builder.GetGraph();
}
/*
 *   netoutput1
 *       |
 *      mul
 *     /   \      \
 * data1   data2  data3
 */
ComputeGraphPtr BuildWhileCondSubGraph3(const std::string name) {
  ut::GraphBuilder builder(name);
  auto data0 = builder.AddNode(name + "data0", "Data", 1, 1);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto mul = builder.AddNode(name + "mul", "Mul", 3, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 1, 1);

  AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", static_cast<int>(2));
  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));

  builder.AddDataEdge(data0, 0, mul, 0);
  builder.AddDataEdge(data1, 0, mul, 1);
  builder.AddDataEdge(data2, 0, mul, 2);
  builder.AddDataEdge(mul, 0, netoutput, 0);

  return builder.GetGraph();
}
/*
 *   netoutput relu
 *       |    /
 *      while
 *     /     \     \
 * data1   data2   const
 */
ComputeGraphPtr BuildMainGraphWithWhile3() {
  ut::GraphBuilder builder("main_graph");
  auto data0 = builder.AddNode("data0", "Data", 1, 1);
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto n = builder.AddNode("n", "Const", 1, 1);
  auto while1 = builder.AddNode("while1", "While", 3, 3);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 2);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);

  builder.AddDataEdge(data0, 0, while1, 0);
  builder.AddDataEdge(data1, 0, while1, 1);
  builder.AddDataEdge(n, 0, while1, 2);
  builder.AddDataEdge(while1, 0, netoutput1, 0);
  builder.AddDataEdge(while1, 1, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildWhileCondSubGraph3("sub1");
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildWhileBodySubGraph3("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("while1"));
  main_graph->FindNode("while1")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("while1")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}
// Check result
void CheckResult(RefRelations &ref_builder, vector<RefCell> &keys, unordered_set<string> &values) {
  for (const auto &key : keys) {
    std::unordered_set<RefCell, RefCellHash>  result;
    auto status = ref_builder.LookUpRefRelations(key, result);
    EXPECT_EQ(status, GRAPH_SUCCESS);
    for (const auto &it : result) {
      string res = it.node_name + std::to_string(it.in_out) + std::to_string(it.in_out_idx) + std::to_string((unsigned long)it.node.get());
      auto iter = values.find(res);
      bool is_exist = (iter == values.end()) ? false : true;
      EXPECT_EQ(is_exist, true);
    }
  }
}
}

TEST_F(UTTEST_RefRelations, Pass_if_1) {
  auto main_graph = BuildMainGraphWithIf();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto if1 = main_graph->FindNode("if");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string if1_s = std::to_string((unsigned long)if1.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);


  vector<RefCell> keys_1 = {
    RefCell("sub1data1",sub1data1, NODE_IN,0),
    RefCell("sub1data1",sub1data1, NODE_OUT,0),
    RefCell("sub2data1",sub2data1, NODE_IN,0),
    RefCell("sub2data1",sub2data1, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 0),
  };

  unordered_set<string> values_1 = {
    string("sub1data100") + sub1data1_s,
    string("sub1data110") + sub1data1_s,
    string("sub2data100") + sub2data1_s,
    string("sub2data110") + sub2data1_s,
    string("if00") + if1_s
  };

  vector<RefCell> keys_2 = {
    RefCell("sub1data2", sub1data2, NODE_IN,0),
    RefCell("sub1data2", sub1data2, NODE_OUT,0),
    RefCell("sub2data2", sub2data2, NODE_IN,0),
    RefCell("sub2data2", sub2data2, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 1),
  };
  unordered_set<string> values_2 = {
    string("sub1data200") + sub1data2_s,
    string("sub1data210") + sub1data2_s,
    string("sub2data200") + sub2data2_s,
    string("sub2data210") + sub2data2_s,
    string("if01") + if1_s
  };

   vector<RefCell> keys_3 = {
    RefCell("sub1netoutput",sub1netoutput, NODE_IN,0),
    RefCell("sub2netoutput",sub2netoutput, NODE_IN,0),
    RefCell("if", if1, NODE_OUT, 0),
  };

  unordered_set<string> values_3 = {
    string("sub1netoutput00") + sub1netoutput_s,
    string("sub2netoutput00") + sub2netoutput_s,
    string("if10") + if1_s
  };

  CheckResult(ref_builder, keys_1, values_1);
  CheckResult(ref_builder, keys_2, values_2);
  CheckResult(ref_builder, keys_3, values_3);

}

TEST_F(UTTEST_RefRelations, Pass_if_2) {
  auto main_graph = BuildMainGraphWithIf2();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto if1 = main_graph->FindNode("if");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string if1_s = std::to_string((unsigned long)if1.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);

  vector<RefCell> keys_1 = {
    RefCell("sub1data1", sub1data1, NODE_IN,0),
    RefCell("sub1data1", sub1data1, NODE_OUT,0),
    RefCell("sub2data1", sub2data1, NODE_IN,0),
    RefCell("sub2data1", sub2data1, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 0),
  };
  unordered_set<string> values_1 = {
    string("sub1data100") + sub1data1_s,
    string("sub1data110") + sub1data1_s,
    string("sub2data100") + sub2data1_s,
    string("sub2data110") + sub2data1_s,
    string("if00") + if1_s
  };

  vector<RefCell> keys_2 = {
    RefCell("sub1data2", sub1data2 ,NODE_IN,0),
    RefCell("sub1data2", sub1data2 ,NODE_OUT,0),
    RefCell("sub2data2", sub2data2 ,NODE_IN,0),
    RefCell("sub2data2", sub2data2 ,NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 1),
  };
  unordered_set<string> values_2 = {
    string("sub1data200") + sub1data2_s,
    string("sub1data210") + sub1data2_s,
    string("sub2data200") + sub2data2_s,
    string("sub2data210") + sub2data2_s,
    string("if01") + if1_s
  };

   vector<RefCell> keys_3 = {
    RefCell("sub1netoutput", sub1netoutput,NODE_IN,0),
    RefCell("sub2netoutput", sub2netoutput,NODE_IN,0),
    RefCell("if", if1,  NODE_OUT, 0),
  };

  unordered_set<string> values_3 = {
    string("sub1netoutput00") + sub1netoutput_s,
    string("sub2netoutput00") + sub2netoutput_s,
    string("if10") + if1_s
  };

  vector<RefCell> keys_4 = {
    RefCell("sub1netoutput",sub1netoutput, NODE_IN,1),
    RefCell("sub2netoutput",sub2netoutput, NODE_IN,1),
    RefCell("if", if1, NODE_OUT, 1),
  };

  unordered_set<string> values_4 = {
    string("sub1netoutput01") + sub1netoutput_s,
    string("sub2netoutput01") + sub2netoutput_s,
    string("if11") + if1_s
  };

  CheckResult(ref_builder, keys_1, values_1);
  CheckResult(ref_builder, keys_2, values_2);
  CheckResult(ref_builder, keys_3, values_3);
  CheckResult(ref_builder, keys_4, values_4);

}

TEST_F(UTTEST_RefRelations, Pass_if_3) {
  auto main_graph = BuildMainGraphWithIf3();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto if1 = main_graph->FindNode("if");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub1data3 = sub1->FindNode("sub1data3");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub2data3 = sub2->FindNode("sub2data3");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string if1_s = std::to_string((unsigned long)if1.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub1data3_s = std::to_string((unsigned long)sub1data3.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub2data3_s = std::to_string((unsigned long)sub2data3.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);


  vector<RefCell> keys_1 = {
    RefCell("sub1data1",sub1data1, NODE_IN,0),
    RefCell("sub1data1",sub1data1, NODE_OUT,0),
    RefCell("sub2data1",sub2data1, NODE_IN,0),
    RefCell("sub2data1",sub2data1, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 0),
  };

  unordered_set<string> values_1 = {
    string("sub1data100") + sub1data1_s,
    string("sub1data110") + sub1data1_s,
    string("sub2data100") + sub2data1_s,
    string("sub2data110") + sub2data1_s,
    string("if00") + if1_s
  };

  vector<RefCell> keys_2 = {
    RefCell("sub1data2", sub1data2, NODE_IN,0),
    RefCell("sub1data2", sub1data2, NODE_OUT,0),
    RefCell("sub2data2", sub2data2, NODE_IN,0),
    RefCell("sub2data2", sub2data2, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 1),
  };
  unordered_set<string> values_2 = {
    string("sub1data200") + sub1data2_s,
    string("sub1data210") + sub1data2_s,
    string("sub2data200") + sub2data2_s,
    string("sub2data210") + sub2data2_s,
    string("if01") + if1_s
  };

  vector<RefCell> keys_4 = {
    RefCell("sub1data3", sub1data3, NODE_IN,0),
    RefCell("sub1data3", sub1data3, NODE_OUT,0),
    RefCell("sub2data3", sub2data3, NODE_IN,0),
    RefCell("sub2data3", sub2data3, NODE_OUT,0),
    RefCell("if", if1, NODE_IN, 1),
  };
  unordered_set<string> values_4 = {
    string("sub1data300") + sub1data3_s,
    string("sub1data310") + sub1data3_s,
    string("sub2data300") + sub2data3_s,
    string("sub2data310") + sub2data3_s,
    string("if01") + if1_s
  };

  vector<RefCell> keys_3 = {
    RefCell("sub1netoutput",sub1netoutput, NODE_IN,0),
    RefCell("sub2netoutput",sub2netoutput, NODE_IN,0),
    RefCell("if", if1, NODE_OUT, 0),
  };

  unordered_set<string> values_3 = {
    string("sub1netoutput00") + sub1netoutput_s,
    string("sub2netoutput00") + sub2netoutput_s,
    string("if10") + if1_s
  };

  CheckResult(ref_builder, keys_1, values_1);
  CheckResult(ref_builder, keys_2, values_2);
  CheckResult(ref_builder, keys_3, values_3);

}

TEST_F(UTTEST_RefRelations, Pass_while) {
  auto main_graph = BuildMainGraphWithWhile();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto while1 = main_graph->FindNode("while1");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub1data3 = sub1->FindNode("sub1data3");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub2data3 = sub2->FindNode("sub2data3");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string while1_s = std::to_string((unsigned long)while1.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub1data3_s = std::to_string((unsigned long)sub1data3.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub2data3_s = std::to_string((unsigned long)sub2data3.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);

  vector<RefCell> keys_1 = {
    RefCell("sub1data1", sub1data1, NODE_IN,0),
    RefCell("sub1data1", sub1data1, NODE_OUT,0),
    RefCell("sub2data1", sub2data1, NODE_IN,0),
    RefCell("sub2data1", sub2data1, NODE_OUT,0),
    RefCell("sub2netoutput", sub2netoutput, NODE_IN,0),
    RefCell("while1", while1, NODE_IN, 0),
    RefCell("while1", while1, NODE_OUT, 0),
  };
  unordered_set<string> values_1 = {
    string("sub1data100") + sub1data1_s,
    string("sub1data110") + sub1data1_s,
    string("sub2data100") + sub2data1_s,
    string("sub2data110") + sub2data1_s,
    string("sub2netoutput00") + sub2netoutput_s,
    string("while100") + while1_s,
    string("while110") + while1_s
  };

  vector<RefCell> keys_2 = {
    RefCell("sub1data2", sub1data2, NODE_IN,0),
    RefCell("sub1data2", sub1data2, NODE_OUT,0),
    RefCell("sub2data2", sub2data2, NODE_IN,0),
    RefCell("sub2data2", sub2data2, NODE_OUT,0),
    RefCell("sub2netoutput", sub2netoutput, NODE_IN,1),
    RefCell("while1", while1, NODE_IN, 1),
    RefCell("while1", while1, NODE_OUT, 1),
  };
  unordered_set<string> values_2 = {
    string("sub1data200")+ sub1data2_s,
    string("sub1data210")+ sub1data2_s,
    string("sub2data200")+ sub2data2_s,
    string("sub2data210")+ sub2data2_s,
    string("sub2netoutput01")+ sub2netoutput_s,
    string("while101")+ while1_s,
    string("while111")+ while1_s
  };

  vector<RefCell> keys_3 = {
    RefCell("sub1data3", sub1data3,NODE_IN,0),
    RefCell("sub1data3", sub1data3,NODE_OUT,0),
    RefCell("sub2data3", sub2data3,NODE_IN,0),
    RefCell("sub2data3", sub2data3,NODE_OUT,0),
    RefCell("sub2netoutput", sub2netoutput,NODE_IN,2),
    RefCell("while1", while1,NODE_IN, 2),
    RefCell("while1", while1,NODE_OUT, 2),
  };
  unordered_set<string> values_3 = {
    string("sub1data300")+ sub1data3_s,
    string("sub1data310")+ sub1data3_s,
    string("sub2data300")+ sub2data3_s,
    string("sub2data310")+ sub2data3_s,
    string("sub2netoutput02")+ sub2netoutput_s,
    string("while102")+ while1_s,
    string("while112")+ while1_s
  };
  CheckResult(ref_builder, keys_1, values_1);
  CheckResult(ref_builder, keys_2, values_2);
  CheckResult(ref_builder, keys_3, values_3);
}

TEST_F(UTTEST_RefRelations, Pass_while_2) {
  auto main_graph = BuildMainGraphWithWhile2();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto while1 = main_graph->FindNode("while1");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub1data3 = sub1->FindNode("sub1data3");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub2data3 = sub2->FindNode("sub2data3");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string while1_s = std::to_string((unsigned long)while1.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub1data3_s = std::to_string((unsigned long)sub1data3.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub2data3_s = std::to_string((unsigned long)sub2data3.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  ref_builder.Clear();
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);

  vector<RefCell> keys_1 = {
    RefCell("sub1data1",sub1data1, NODE_IN,0),
    RefCell("sub1data1",sub1data1 ,NODE_OUT,0),
    RefCell("sub2data1",sub2data1 ,NODE_IN,0),
    RefCell("sub2data1",sub2data1 ,NODE_OUT,0),
    RefCell("sub2netoutput",sub2netoutput ,NODE_IN,1),
    RefCell("while1",while1 ,NODE_IN, 0),
    RefCell("while1",while1 ,NODE_OUT, 0),
  };
  unordered_set<string> values_1 = {
    string("sub1data100") + sub1data1_s,
    string("sub1data110") + sub1data1_s,
    string("sub2data100") + sub2data1_s,
    string("sub2data110") + sub2data1_s,
    string("sub2netoutput01") + sub2netoutput_s,
    string("while100") + while1_s,
    string("while110") + while1_s
  };

  // vector<RefCell> keys_2 = {
  //   RefCell("sub1data2",sub1data2 ,NODE_IN,0),
  //   RefCell("sub1data2",sub1data2 ,NODE_OUT,0),
  //   RefCell("sub2data2",sub2data2 ,NODE_IN,0),
  //   RefCell("sub2data2",sub2data2 ,NODE_OUT,0),
  //   RefCell("sub2netoutput",sub2netoutput,NODE_IN,0),
  //   RefCell("sub2netoutput",sub2netoutput ,NODE_OUT,0),
  //   RefCell("while1",while1 ,NODE_IN, 1),
  //   RefCell("while1",while1 ,NODE_OUT, 1),
  // };
  // unordered_set<string> values_2 = {
  //   string("sub1data200")+ sub1data2_s,
  //   string("sub1data210")+ sub1data2_s,
  //   string("sub2data200")+ sub2data2_s,
  //   string("sub2data210")+ sub2data2_s,
  //   string("sub2netoutput00")+ sub2netoutput_s,
  //   string("sub2netoutput10")+ sub2netoutput_s,
  //   string("while101")+ while1_s,
  //   string("while111")+ while1_s
  // };


  // vector<RefCell> keys_3 = {
  //   RefCell("sub1data3",sub1data3 ,NODE_IN,0),
  //   RefCell("sub1data3",sub1data3 ,NODE_OUT,0),
  //   RefCell("sub2data3",sub2data3 ,NODE_IN,0),
  //   RefCell("sub2data3",sub2data3 ,NODE_OUT,0),
  //   RefCell("sub2netoutput",sub2netoutput ,NODE_IN,2),
  //   RefCell("sub2netoutput",sub2netoutput ,NODE_OUT,2),
  //   RefCell("while1",while1 ,NODE_IN, 2),
  //   RefCell("while1",while1 ,NODE_OUT, 2),
  // };
  // unordered_set<string> values_3 = {
  //   string("sub1data300")+ sub1data3_s,
  //   string("sub1data310")+ sub1data3_s,
  //   string("sub2data300")+ sub2data3_s,
  //   string("sub2data310")+ sub2data3_s,
  //   string("sub2netoutput02")+ sub2netoutput_s,
  //   string("sub2netoutput12")+ sub2netoutput_s,
  //   string("while102")+ while1_s,
  //   string("while112")+ while1_s
  // };

  CheckResult(ref_builder, keys_1, values_1);
  // CheckResult(ref_builder, keys_2, values_2);
  // CheckResult(ref_builder, keys_3, values_3);
}

TEST_F(UTTEST_RefRelations, Pass_while3) {
  auto main_graph = BuildMainGraphWithWhile3();

  auto sub1 = main_graph->GetSubgraph("sub1");
  auto sub2 = main_graph->GetSubgraph("sub2");
  auto while1 = main_graph->FindNode("while1");
  auto sub1data0 = sub1->FindNode("sub1data0");
  auto sub1data1 = sub1->FindNode("sub1data1");
  auto sub1data2 = sub1->FindNode("sub1data2");
  auto sub2data0 = sub2->FindNode("sub2data0");
  auto sub2data1 = sub2->FindNode("sub2data1");
  auto sub2data2 = sub2->FindNode("sub2data2");
  auto sub1netoutput = sub1->FindNode("sub1netoutput");
  auto sub2netoutput = sub2->FindNode("sub2netoutput");

  string while1_s = std::to_string((unsigned long)while1.get());
  string sub1data0_s = std::to_string((unsigned long)sub1data0.get());
  string sub1data1_s = std::to_string((unsigned long)sub1data1.get());
  string sub1data2_s = std::to_string((unsigned long)sub1data2.get());
  string sub2data0_s = std::to_string((unsigned long)sub2data0.get());
  string sub2data1_s = std::to_string((unsigned long)sub2data1.get());
  string sub2data2_s = std::to_string((unsigned long)sub2data2.get());
  string sub1netoutput_s = std::to_string((unsigned long)sub1netoutput.get());
  string sub2netoutput_s = std::to_string((unsigned long)sub2netoutput.get());

  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_NE(status, GRAPH_SUCCESS);

  vector<RefCell> keys_0 = {
    RefCell("sub1data0", sub1data0, NODE_IN,0),
    RefCell("sub1data0", sub1data0, NODE_OUT,0),
    RefCell("sub2data0", sub2data0, NODE_IN,0),
    RefCell("sub2data0", sub2data0, NODE_OUT,0),
    RefCell("sub2netoutput", sub2netoutput, NODE_IN,2),
    RefCell("while1", while1, NODE_IN, 2),
    RefCell("while1", while1, NODE_OUT, 2),
  };
  unordered_set<string> values_0 = {
    string("sub1data000") + sub1data0_s,
    string("sub1data010") + sub1data0_s,
    string("sub2data000") + sub2data0_s,
    string("sub2data010") + sub2data0_s,
    string("sub2netoutput02") + sub2netoutput_s,
    string("while102") + while1_s,
    string("while112") + while1_s
  };

  vector<RefCell> keys_1 = {
    RefCell("sub1data1", sub1data1, NODE_IN,0),
    RefCell("sub1data1", sub1data1, NODE_OUT,0),
    RefCell("sub2data1", sub2data1, NODE_IN,0),
    RefCell("sub2data1", sub2data1, NODE_OUT,0),
    RefCell("sub1data2", sub1data2, NODE_IN,0),
    RefCell("sub1data2", sub1data2, NODE_OUT,0),
    RefCell("sub2data2", sub2data2, NODE_IN,0),
    RefCell("sub2data2", sub2data2, NODE_OUT,0),
    RefCell("sub2netoutput", sub2netoutput, NODE_IN,0),
    RefCell("sub2netoutput", sub2netoutput, NODE_IN,1),
    RefCell("while1", while1, NODE_IN, 0),
    RefCell("while1", while1, NODE_OUT, 0),
    RefCell("while1", while1, NODE_IN, 1),
    RefCell("while1", while1, NODE_OUT, 1),
  };
  unordered_set<string> values_1 = {
    string("sub1data100")+ sub1data1_s,
    string("sub1data110")+ sub1data1_s,
    string("sub2data100")+ sub2data1_s,
    string("sub2data110")+ sub2data1_s,
    string("sub1data200")+ sub1data2_s,
    string("sub1data210")+ sub1data2_s,
    string("sub2data200")+ sub2data2_s,
    string("sub2data210")+ sub2data2_s,
    string("sub2netoutput01")+ sub2netoutput_s,
    string("sub2netoutput00")+ sub2netoutput_s,
    string("while100")+ while1_s,
    string("while110")+ while1_s,
    string("while101")+ while1_s,
    string("while111")+ while1_s
  };

  // CheckResult(ref_builder, keys_0, values_0);
  CheckResult(ref_builder, keys_1, values_1);
}
TEST_F(UTTEST_RefRelations, Failed_if_1) {
  auto main_graph = BuildMainGraphWithIfButWithNoSubgraph();
  RefRelations ref_builder;
  auto status = ref_builder.BuildRefRelations(*main_graph);
  EXPECT_EQ(status, GRAPH_SUCCESS);
}
}