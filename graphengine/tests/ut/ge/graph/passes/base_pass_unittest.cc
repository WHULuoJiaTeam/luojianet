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

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "gtest/gtest.h"

#define protected public
#include "graph/passes/base_pass.h"
#undef protected

#include "framework/common/types.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

template class std::unordered_set<ge::NodePtr>;

namespace ge {
class UtestTestPass : public BaseNodePass {
 public:
  UtestTestPass() = default;
  UtestTestPass(bool dead_loop) : dead_loop_(dead_loop), run_times_(0) {}

  Status Run(NodePtr &node) override {
    ++run_times_;
    iter_nodes_.push_back(node);
    auto iter = names_to_add_del_.find(node->GetName());
    if (iter != names_to_add_del_.end()) {
      for (const auto &node_name : iter->second) {
        auto del_node = node->GetOwnerComputeGraph()->FindNode(node_name);
        GraphUtils::IsolateNode(del_node, {0});
        AddNodeDeleted(del_node);
      }
    }
    iter = names_to_add_repass_.find(node->GetName());
    if (iter != names_to_add_repass_.end()) {
      auto all_nodes = node->GetOwnerComputeGraph()->GetAllNodes();
      for (const auto &node_name : iter->second) {
        for (auto &node_re_pass : all_nodes) {
          if (node_re_pass->GetName() == node_name) {
            AddRePassNode(node_re_pass);
            break;
          }
        }
      }
      if (!dead_loop_) {
        names_to_add_repass_.erase(iter);
      }
    }

    iter = names_to_add_repass_immediate_.find(node->GetName());
    if (iter != names_to_add_repass_immediate_.end()) {
      auto all_nodes = node->GetOwnerComputeGraph()->GetAllNodes();
      for (const auto &node_name : iter->second) {
        for (auto &node_re_pass : all_nodes) {
          if (node_re_pass->GetName() == node_name) {
            AddImmediateRePassNode(node_re_pass);
            break;
          }
        }
      }
      if (!dead_loop_) {
        names_to_add_repass_immediate_.erase(iter);
      }
    }

    iter = names_to_add_suspend_.find(node->GetName());
    if (iter != names_to_add_suspend_.end()) {
      auto all_nodes = node->GetOwnerComputeGraph()->GetAllNodes();
      for (const auto &node_name : iter->second) {
        for (auto &node_re_pass : all_nodes) {
          if (node_re_pass->GetName() == node_name) {
            AddNodeSuspend(node_re_pass);
            break;
          }
        }
      }
      if (!dead_loop_) {
        names_to_add_suspend_.erase(iter);
      }
    }

    iter = names_to_add_resume_.find(node->GetName());
    if (iter != names_to_add_resume_.end()) {
      auto all_nodes = node->GetOwnerComputeGraph()->GetAllNodes();
      for (const auto &node_name : iter->second) {
        for (auto &node_re_pass : all_nodes) {
          if (node_re_pass->GetName() == node_name) {
            AddNodeResume(node_re_pass);
            break;
          }
        }
      }
      if (!dead_loop_) {
        names_to_add_resume_.erase(iter);
      }
    }
    // simulate infershape pass
    if(node->GetType() == WHILE){
      bool need_repass = false;
      AttrUtils::GetBool(node->GetOpDesc(),"_need_infer_again", need_repass);
      if(!OptionExists(kOptimizeAfterSubGraph)){
        return SUCCESS;
      }
      if(need_repass){
        AttrUtils::SetBool(node->GetOpDesc(),"_need_infer_again", false);
        AddImmediateRePassNode(node);
      }
      else{
        // clear attr on while
        node->GetOpDesc()->DelAttr("_need_infer_again");
      }
    }
    return SUCCESS;
  }

  Status OnSuspendNodesLeaked() override {
    // resume all node remain in suspend_nodes when leaked
    auto compute_graph = (iter_nodes_.size() > 0) ? iter_nodes_[0]->GetOwnerComputeGraph() : nullptr;
    if (compute_graph == nullptr) {
      return SUCCESS;
    }

    for (const auto &node_name : names_to_add_resume_onleaked_) {
      auto node_to_resume = compute_graph->FindNode(node_name);
      AddNodeResume(node_to_resume);
    }
    return SUCCESS;
  }
  void clear() { iter_nodes_.clear(); }
  std::vector<NodePtr> GetIterNodes() { return iter_nodes_; }

  void AddRePassNodeName(const std::string &iter_node, const std::string &re_pass_node) {
    names_to_add_repass_[iter_node].insert(re_pass_node);
  }
  void AddDelNodeName(const std::string &iter_node, const std::string &del_node) {
    names_to_add_del_[iter_node].insert(del_node);
  }
  void AddRePassImmediateNodeName(const std::string &iter_node, const std::string &re_pass_node) {
    names_to_add_repass_immediate_[iter_node].insert(re_pass_node);
  }

  void AddSuspendNodeName(const std::string &iter_node, const std::string &suspend_node) {
    names_to_add_suspend_[iter_node].insert(suspend_node);
  }
  void AddResumeNodeName(const std::string &iter_node, const std::string &resume_node) {
    names_to_add_resume_[iter_node].insert(resume_node);
  }
  void AddResumeNodeNameOnLeaked(const std::string &resume_node) {
    names_to_add_resume_onleaked_.insert(resume_node);
  }

  unsigned int GetRunTimes() { return run_times_; }

 private:
  std::vector<NodePtr> iter_nodes_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_del_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_repass_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_repass_immediate_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_suspend_;
  std::map<std::string, std::unordered_set<std::string>> names_to_add_resume_;
  std::unordered_set<std::string> names_to_add_resume_onleaked_;

  bool dead_loop_;
  unsigned int run_times_;
};

class TestDelPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override { return SUCCESS; }
};

class UTESTGraphPassesBasePass : public testing::Test {
 protected:
  UTESTGraphPassesBasePass() {
    auto p1 = new UtestTestPass;
    names_to_pass_.push_back(std::make_pair("test1", p1));
  }
  void SetUp() override {
    for (auto &name_to_pass : names_to_pass_) {
      dynamic_cast<UtestTestPass *>(name_to_pass.second)->clear();
    }
  }
  ~UTESTGraphPassesBasePass() override {
    for (auto &name_to_pass : names_to_pass_) {
      delete name_to_pass.second;
    }
  }
  NamesToPass names_to_pass_;
};
///      reshape1
///        |
///       add1
///     /     \.
///    |      |
///  data1  const1
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto a1 = builder.AddNode("add1", ADD, 2, 1);
  auto c1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto r1 = builder.AddNode("reshape1", RESHAPE, 1, 1);

  builder.AddDataEdge(data, 0, a1, 0);
  builder.AddDataEdge(c1, 0, a1, 1);
  builder.AddDataEdge(a1, 0, r1, 0);

  return builder.GetGraph();
}

///               sum1
///             /     \.
///            /       \.
///          /          \.
///      reshape1      addn1
///        |      c      |
///       add1  <---  shape1
///     /     \         |
///    |      |         |
///  data1  const1    const2
ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto shape1 = builder.AddNode("shape1", SHAPE, 1, 1);
  auto reshape1 = builder.AddNode("reshape1", RESHAPE, 1, 1);
  auto addn1 = builder.AddNode("addn1", ADDN, 1, 1);
  auto sum1 = builder.AddNode("sum1", SUM, 2, 1);

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(const2, 0, shape1, 0);
  builder.AddControlEdge(shape1, add1);
  builder.AddDataEdge(add1, 0, reshape1, 0);
  builder.AddDataEdge(shape1, 0, addn1, 0);
  builder.AddDataEdge(reshape1, 0, sum1, 0);
  builder.AddDataEdge(addn1, 0, sum1, 1);

  return builder.GetGraph();
}

///   rnextiteration
///    |  |
///   merge
///    |
///  data1
ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto merge1 = builder.AddNode("merge1", MERGE, 2, 1);
  auto next1 = builder.AddNode("next1", NEXTITERATION, 1, 1);

  builder.AddDataEdge(data1, 0, merge1, 0);
  builder.AddDataEdge(merge1, 0, next1, 0);
  builder.AddDataEdge(next1, 0, merge1, 1);
  builder.AddControlEdge(merge1, next1);
  builder.AddControlEdge(next1, merge1);

  return builder.GetGraph();
}

///          cast1--shape1
///        /
///  data1
///       \
///        transdata1--shape2
ComputeGraphPtr BuildGraph4() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1);
  auto shape1 = builder.AddNode("shape1", SHAPE, 1, 1);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  auto shape2 = builder.AddNode("shape2", SHAPE, 1, 1);

  builder.AddDataEdge(data1, 0, cast1, 0);
  builder.AddDataEdge(data1, 0, transdata1, 0);
  builder.AddDataEdge(cast1, 0, shape1, 0);
  builder.AddDataEdge(transdata1, 0, shape2, 0);
  return builder.GetGraph();
}

void CheckIterOrder(UtestTestPass *pass, std::vector<std::unordered_set<std::string>> &nodes_layers) {
  std::unordered_set<std::string> layer_nodes;
  size_t layer_index = 0;
  for (const auto &node : pass->GetIterNodes()) {
    layer_nodes.insert(node->GetName());
    EXPECT_LT(layer_index, nodes_layers.size());
    if (layer_nodes == nodes_layers[layer_index]) {
      layer_index++;
      layer_nodes.clear();
    }
  }
  EXPECT_EQ(layer_index, nodes_layers.size());
}

///      Op1
///       |
///     Merge
///      / \.
///    Op2 Op3
TEST_F(UTESTGraphPassesBasePass, del_isolate_fail) {
  auto builder = ut::GraphBuilder("g1");
  auto merge_node = builder.AddNode("Merge", MERGE, 1, 1);
  auto node1 = builder.AddNode("Op1", RELU, 1, 1);
  auto node2 = builder.AddNode("Op2", CONVOLUTION, 1, 1);
  auto node3 = builder.AddNode("Op3", CONVOLUTION, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  TestDelPass del_pass;
  auto ret = del_pass.IsolateAndDeleteNode(merge_node, {0, -1});
  EXPECT_EQ(ret, FAILED);

  OpDescPtr op_desc = std::make_shared<OpDesc>("merge", MERGE);
  NodePtr node = shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));
  ret = del_pass.IsolateAndDeleteNode(node, {0, -1});
  EXPECT_EQ(ret, FAILED);
}

///      Op1
///       |
///     Merge
///      / \.
///    Op2 Op3
TEST_F(UTESTGraphPassesBasePass, del_isolate_success) {
  auto builder = ut::GraphBuilder("g1");
  auto merge_node = builder.AddNode("Merge", MERGE, 1, 2);
  auto node1 = builder.AddNode("Op1", RELU, 1, 1);
  auto node2 = builder.AddNode("Op2", CONVOLUTION, 1, 1);
  auto node3 = builder.AddNode("Op3", CONVOLUTION, 1, 1);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), merge_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  EXPECT_EQ(node1->GetOutDataNodes().size(), 1);

  TestDelPass del_pass;
  auto ret = del_pass.IsolateAndDeleteNode(merge_node, {0, -1});
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UTESTGraphPassesBasePass, data_graph) {
  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  auto *pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);

  EXPECT_EQ(pass->GetIterNodes().size(), 4);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1"});
  layers.push_back({"add1"});
  layers.push_back({"reshape1"});
  CheckIterOrder(pass, layers);
}

TEST_F(UTESTGraphPassesBasePass, graph_with_control_link) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  auto *pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);

  EXPECT_EQ(pass->GetIterNodes().size(), 8);
  EXPECT_EQ(pass->GetIterNodes().at(3)->GetName(), "shape1");

  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1", "reshape1"});
  layers.push_back({"sum1"});
  CheckIterOrder(pass, layers);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_after) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddRePassNodeName("shape1", "sum1");
  test_pass.AddRePassNodeName("shape1", "add1");
  test_pass.AddRePassNodeName("data1", "add1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 8);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_before) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "data1");

  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 5);
  EXPECT_EQ(test_pass.GetIterNodes().at(2)->GetName(), "add1");
  EXPECT_EQ(test_pass.GetIterNodes().at(3)->GetName(), "reshape1");
  EXPECT_EQ(test_pass.GetIterNodes().at(4)->GetName(), "data1");
}

TEST_F(UTESTGraphPassesBasePass, re_pass_before_multi_times) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "data1");
  test_pass.AddRePassNodeName("add1", "const1");
  test_pass.AddRePassNodeName("reshape1", "data1");

  auto graph = BuildGraph1();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 6);
  EXPECT_EQ(test_pass.GetIterNodes().at(2)->GetName(), "add1");
  EXPECT_EQ(test_pass.GetIterNodes().at(3)->GetName(), "reshape1");
}

TEST_F(UTESTGraphPassesBasePass, del_after) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("add1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}

TEST_F(UTESTGraphPassesBasePass, del_after_multiple) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("add1", "sum1");
  test_pass.AddDelNodeName("add1", "reshape1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 6);
}

TEST_F(UTESTGraphPassesBasePass, del_after_break_link) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("shape1", "add1");
  test_pass.AddDelNodeName("shape1", "addn1");
  test_pass.AddRePassNodeName("shape1", "shape1");
  test_pass.AddRePassNodeName("shape1", "reshape1");
  test_pass.AddRePassNodeName("shape1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}

TEST_F(UTESTGraphPassesBasePass, del_self_and_after) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("shape1", "add1");
  test_pass.AddDelNodeName("shape1", "addn1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 6);
}

TEST_F(UTESTGraphPassesBasePass, del_before) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddDelNodeName("reshape1", "add1");
  test_pass.AddDelNodeName("sum1", "addn1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 8);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_and_del) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddDelNodeName("reshape1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 7);
}
/*
TEST_F(UTESTGraphPassesBasePass, dead_loop) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass(true);
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  test_pass.AddRePassNodeName("add1", "sum1");
  test_pass.AddRePassNodeName("sum1", "add1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetRunTimes(), 1007);
}
*/

TEST_F(UTESTGraphPassesBasePass, while_loop) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass(true);
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  auto graph = BuildGraph3();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
}

///     data1  const
///        \ /
///       while
///     /     \.
///    |      |
///   cast1   cast2
ComputeGraphPtr BuildWhileGraph1() {
  // build sub graph
  auto builder_sub = ut::GraphBuilder("sub");
  auto data_1 = builder_sub.AddNode("data_1", DATA, 0, 1);
  auto data_2 = builder_sub.AddNode("data_2", DATA, 0, 1);
  auto add =  builder_sub.AddNode("add", ADD, 2, 1);

  builder_sub.AddDataEdge(data_1, 0, add, 0);
  builder_sub.AddDataEdge(data_2, 0, add, 1);
  auto sub_graph = builder_sub.GetGraph();
  sub_graph->SetName("while_sub");
  // build root graph
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data1", DATA, 0, 1);
  auto const_op = builder.AddNode("const_op", CONSTANT, 0, 1);
  auto c1 = builder.AddNode("cast1", CAST, 1, 1);
  auto c2 = builder.AddNode("cast2", CAST, 1, 1);
  // add while op
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1,1,1,1}));
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetDataType(DT_INT32);

  auto op_desc = std::make_shared<OpDesc>("while", WHILE);
  for (int i = 0; i < 2; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < 2; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  AttrUtils::SetBool(op_desc,"_need_infer_again", true);
  op_desc->AddSubgraphName(sub_graph->GetName());
  op_desc->SetSubgraphInstanceName(0,sub_graph->GetName());
  auto root_graph = builder.GetGraph();
  auto while_op = root_graph->AddNode(op_desc);

  builder.AddDataEdge(data, 0, while_op, 0);
  builder.AddDataEdge(const_op, 0, while_op, 1);
  builder.AddDataEdge(while_op, 0, c1, 0);
  builder.AddDataEdge(while_op, 1, c2, 0);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(while_op);
  root_graph->AddSubgraph(sub_graph);
  return root_graph;
}

TEST_F(UTESTGraphPassesBasePass, while_infershape) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  auto graph = BuildWhileGraph1();
  auto ge_pass = GEPass(graph);
  auto while_node = graph->FindNode("while");
  EXPECT_EQ(while_node->GetOpDesc()->GetSubgraphInstanceNames().size(),1);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_pre_node_immediately) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // repass pre_node immediately
  test_pass->AddRePassImmediateNodeName("reshape1", "add1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);

  EXPECT_EQ(test_pass->GetIterNodes().size(), 9);// todo
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "add1", "sum1"});
  CheckIterOrder(test_pass, layers);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_cur_node_immediately) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // repass cur_node immediately
  test_pass->AddRePassImmediateNodeName("reshape1", "reshape1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);

  EXPECT_EQ(test_pass->GetIterNodes().size(), 9);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1"});
  layers.push_back({"reshape1", "sum1"});
  CheckIterOrder(test_pass, layers);
}

TEST_F(UTESTGraphPassesBasePass, re_pass_next_node_immediately) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // repass next_node immediately
  test_pass->AddRePassImmediateNodeName("reshape1", "sum1");
  // repass node after next_node immediately
  test_pass->AddRePassImmediateNodeName("add1", "sum1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);

  EXPECT_EQ(test_pass->GetIterNodes().size(), 8);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1"});
  CheckIterOrder(test_pass, layers);
}
/**
 * A->B->C
 * if node B suspend its pre_node A, and C resume A,  it is a useless operation, so iter_order should follow normal order
 * when C resuem A, A will pass again.
 */
TEST_F(UTESTGraphPassesBasePass, B_suspend_pre_node_A_then_C_resume_A) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // add1->reshape1->sum1
  test_pass->AddSuspendNodeName("reshape1", "add1");
  test_pass->AddResumeNodeName("sum1", "add1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 9);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1"});
  layers.push_back({"add1"});
  CheckIterOrder(test_pass, layers);
}

/**
 * A->B->C
 * if node B suspend its pre_node A, and B resume A,  it is a useless operation, so iter_order should follow normal order
 * when B resuem A, A will pass again.
 */
TEST_F(UTESTGraphPassesBasePass, B_suspend_pre_node_A_then_B_resume_A) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // add1->reshape1->sum1
  test_pass->AddSuspendNodeName("reshape1", "add1");
  test_pass->AddResumeNodeName("reshape1", "add1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 9);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1", "add1"});
  CheckIterOrder(test_pass, layers);
}

/**
 * A->B->C
 * if node B resume C(which is not suspended),  it is a useless operation, C will not pass.
 */
TEST_F(UTESTGraphPassesBasePass, B_resume_node_not_suspended) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // add1->reshape1->sum1
  test_pass->AddResumeNodeName("reshape1", "sum1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 8);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1"});
  CheckIterOrder(test_pass, layers);
}

/**
 * A->B->C
 * if node B suspend its pre_node A, it is a useless operation, so iter_order should follow normal order
 * because nobody resume it ,which means A is a leaked node, so return fail
 */
TEST_F(UTESTGraphPassesBasePass, suspend_pre_node_nobody_resume_it_return_failed) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));
  // suspend pre_node immediately
  test_pass.AddSuspendNodeName("reshape1", "add1");
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), INTERNAL_ERROR);
}

/**
 * A->B->C
 * if node B suspend its pre_node A, it is a useless operation,
 * so iter_order should follow normal order
 * resume A on leaked, which means A will pass again
 */
TEST_F(UTESTGraphPassesBasePass, suspend_pre_node_resume_it_onleaked) {
  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("reshape1", "add1");
  test_pass->AddResumeNodeNameOnLeaked("add1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1"});
  layers.push_back({"add1"});
  CheckIterOrder(test_pass, layers);
}


///          cast1--shape1
///        /
///  data1
///       \
///        transdata1--shape2
/**
 * suspend cur node
 * cast1 suspend itself, shape2 resume cast1
 * iter order follows : data1; cast1,transdata1; shape2; cast1 ; shape1
 */
TEST_F(UTESTGraphPassesBasePass, cast1_suspend_cur_node_shape2_resume_cast1) {
  auto graph = BuildGraph4();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("cast1", "cast1");
  test_pass->AddResumeNodeName("shape2", "cast1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 6);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1"});
  layers.push_back({"cast1","transdata1"});
  layers.push_back({"shape2"});
  layers.push_back({"cast1", "shape1"});
  CheckIterOrder(test_pass, layers);
}
/**
 * suspend cur node
 * cast1 suspend itself, then resume cast1
 * iter order follows : data1; cast1,cast1,transdata1; shape2; shape1.
 */
TEST_F(UTESTGraphPassesBasePass, cast1_suspend_itslef_then_resume_itself) {
  auto graph = BuildGraph4();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("cast1", "cast1");
  test_pass->AddResumeNodeName("cast1", "cast1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 6);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1"});
  layers.push_back({"cast1","transdata1","cast1","shape1", "shape2"});
  CheckIterOrder(test_pass, layers);
}
/**
 * suspend cur node
 * cast1 suspend itself, then resume cast1 on leaked
 * iter order follows : data1; cast1,cast1,transdata1; shape2; shape1.
 */
TEST_F(UTESTGraphPassesBasePass, cast1_suspend_itslef_then_resume_onleaked) {
  auto graph = BuildGraph4();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("cast1", "cast1");
  test_pass->AddResumeNodeNameOnLeaked("cast1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 6);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1"});
  layers.push_back({"cast1","transdata1", "shape2"});
  layers.push_back({"cast1","shape1"});
  CheckIterOrder(test_pass, layers);
}
/**
 * suspend next node
 * data1 suspend cast1, then resume cast1 on leaked
 * iter order follows : data1; transdata1, shape2; cast1, shape1.
 */
TEST_F(UTESTGraphPassesBasePass, data1_suspend_cast1_resume_cast1_onleaked) {
  auto graph = BuildGraph4();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("data1", "cast1");
  test_pass->AddResumeNodeNameOnLeaked("cast1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), SUCCESS);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 5);
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1"});
  layers.push_back({"transdata1", "shape2"});
  layers.push_back({"cast1","shape1"});
  CheckIterOrder(test_pass, layers);
}

/**
 * suspend next node
 * data1 suspend cast1, nobody resume it
 * iter order follows : data1; transdata1, shape2;
 * run ret is failed ,because node leaked
 */
TEST_F(UTESTGraphPassesBasePass, data1_suspend_cast1_nobody_resume) {
  auto graph = BuildGraph4();
  auto ge_pass = GEPass(graph);
  auto *test_pass = dynamic_cast<UtestTestPass *>(names_to_pass_[0].second);
  // suspend pre_node immediately
  test_pass->AddSuspendNodeName("data1", "cast1");
  EXPECT_EQ(ge_pass.Run(names_to_pass_), INTERNAL_ERROR);
  EXPECT_EQ(test_pass->GetIterNodes().size(), 3);
}

/*
TEST_F(UTESTGraphPassesBasePass, suspend_pre_node) {
  NamesToPass names_to_pass;
  auto test_pass = UtestTestPass();
  names_to_pass.push_back(std::make_pair("test", &test_pass));

  // repass next_node immediately
  test_pass.AddRePassNodeName("reshape1", "sum1");
  // repass node after next_node immediately
  test_pass.AddRePassNodeName("add1", "sum1");

  auto graph = BuildGraph2();
  auto ge_pass = GEPass(graph);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(test_pass.GetIterNodes().size(), 8);// todo
  std::vector<std::unordered_set<std::string>> layers;
  layers.push_back({"data1", "const1", "const2"});
  layers.push_back({"shape1"});
  layers.push_back({"add1", "addn1"});
  layers.push_back({"reshape1", "sum1"});
  CheckIterOrder(&test_pass, layers);
}*/
}  // namespace ge
