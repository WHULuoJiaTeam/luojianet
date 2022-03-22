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

#include "graph/passes/infer_base_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

using namespace std;
using namespace testing;
namespace ge {
class ChildPassBuilder;
static const char *kInferTimes = "infer_times";
class InferBasePassStub : public InferBasePass {
 public:
  friend class ChildPassBuilder;
  graphStatus Infer(NodePtr &node) override{
    call_infer_times++;
    for (size_t i = 0; i < node->GetOutDataNodesSize(); ++i) {
      auto output_td = node->GetOpDesc()->MutableOutputDesc(i);
      int times = 0;
      AttrUtils::GetInt(output_td, kInferTimes, times);
      AttrUtils::SetInt(output_td, kInferTimes, times + 1);
    }
    return infer_result_;
  };

  int32_t call_infer_times = 0;
  int32_t call_update_tensor_desc_times = 0;
  int32_t call_update_from_subgraph_times = 0;
  int32_t call_update_from_subgraph_multi_dims_times = 0;
  std::vector<std::pair<GeTensorDescPtr, GeTensorDescPtr>> update_td_pairs;

 private:
  bool NeedInfer(const NodePtr &node) const override {
    return need_infer_;
  };
  std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const override { return "test SerialTensorInfo"; };
  graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) override {
    call_update_tensor_desc_times++;
    changed = td_changed_;
    int times = 0;
    if (AttrUtils::GetInt(src, kInferTimes, times)) {
      AttrUtils::SetInt(dst, kInferTimes, times);
    }
    update_td_pairs.emplace_back(src, dst);
    return GRAPH_SUCCESS;
  };
  graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src, GeTensorDescPtr &dst) override {
    call_update_from_subgraph_times++;
    return GRAPH_SUCCESS;
  };
  graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                    GeTensorDescPtr &dst) override {
    call_update_from_subgraph_multi_dims_times++;
    return GRAPH_SUCCESS;
  };
  bool td_changed_;
  bool need_infer_;
  graphStatus infer_result_;
};

class ChildPassBuilder {
 public:
  ChildPassBuilder &SetNeedInferFlag(bool flag) {
    need_infer_ = flag;
    return *this;
  }

  ChildPassBuilder &SetInferResult(graphStatus ret) {
    infer_result_ = ret;
    return *this;
  }

  ChildPassBuilder &SetTdChangedFlag(bool changed_flag) {
    td_changed_ = changed_flag;
    return *this;
  }

  InferBasePassStub Build() {
    InferBasePassStub ib;
    ib.td_changed_ = td_changed_;
    ib.need_infer_ = need_infer_;
    ib.infer_result_ = infer_result_;
    return ib;
  }

 private:
  bool td_changed_ = false;
  bool need_infer_ = true;
  graphStatus infer_result_ = GRAPH_SUCCESS;
};

class UtestGraphInferBasePassStub : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *   data1   data2
 *      \     /
 *        sub1
 *         |
 *     netoutput
 */
ut::GraphBuilder TestSubgraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("branch_graph");
  std::vector<int64_t> shape1 = {1,1};
  auto data1 = builder.AddNode("data1_1", "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape1);
  auto data1_desc = data1->GetOpDesc();
  EXPECT_NE(data1_desc, nullptr);
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);
  std::vector<int64_t> shape2 = {2,2};
  auto data2 = builder.AddNode("data2_1", "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape2);
  auto data2_desc = data2->GetOpDesc();
  EXPECT_NE(data2_desc, nullptr);
  AttrUtils::SetInt(data2_desc, "_parent_node_index", 1);

  auto sub1 = builder.AddNode("Sub", "Sub", 2, 1);
  std::vector<int64_t> shape7 = {8,8};
  auto netoutput = builder.AddNode("output", NETOUTPUT, 1, 0, FORMAT_NCHW, DT_INT32, shape7);
  auto input0_desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  EXPECT_NE(input0_desc, nullptr);
  AttrUtils::SetInt(input0_desc, "_parent_node_index", 0);

  builder.AddDataEdge(data1, 0, sub1, 0);
  builder.AddDataEdge(data2, 0, sub1, 1);
  builder.AddDataEdge(sub1, 0, netoutput, 0);
  return builder;
}

/*
 *   data1  data2
 *     \     /
 *      case1
 *        |
 *    netoutput
 */
ut::GraphBuilder RootGraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("root_graph");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(data2, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput, 0);

  auto parent_graph = builder.GetGraph();
  auto subgraph_builder = TestSubgraphBuilder();
  auto subgraph = subgraph_builder.GetGraph();
  case1->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  case1->GetOpDesc()->SetSubgraphInstanceName(0, subgraph->GetName());
  subgraph->SetParentNode(case1);
  subgraph->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph(subgraph->GetName(), subgraph), GRAPH_SUCCESS);
  return builder;
}

/*
 *   data1   data2
 *      \     /
 *       add1
 *         |
 *     netoutput
 */
ut::GraphBuilder NoSubgraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("no_subgraph");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(data2, 0, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput, 0);
  return builder;
}

TEST_F(UtestGraphInferBasePassStub, CallInfer_WhenNeedInferReturnTrue) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  EXPECT_NE(add_node, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.Build();

  // NeedInfer return true
  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_infer_times, 1);
  int times = -1;
  EXPECT_TRUE(AttrUtils::GetInt(add_node->GetOpDesc()->GetOutputDescPtr(0), kInferTimes, times));
  EXPECT_EQ(times, 1);
}

TEST_F(UtestGraphInferBasePassStub, NotCallInfer_WhenNeedInferReturnFalse) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  EXPECT_NE(add_node, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.SetNeedInferFlag(false).Build();

  // NeedInfer return false
  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_infer_times, 0);
  int times = -1;
  EXPECT_FALSE(AttrUtils::GetInt(add_node->GetOpDesc()->GetOutputDescPtr(0), kInferTimes, times));
}

TEST_F(UtestGraphInferBasePassStub, NotAddCurNodeRepass_CallUpdatePeerNode_WhenInferReturnSuccess) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  auto netoutput = test_graph->FindNode("netoutput");
  EXPECT_NE(add_node, nullptr);
  EXPECT_NE(netoutput, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.Build();

  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_infer_times, 1);
  EXPECT_EQ(stub_base_pass.call_update_tensor_desc_times, 1);
  std::vector<std::pair<GeTensorDescPtr, GeTensorDescPtr>> expected_updated_tensor_desc_pairs = {
    {add_node->GetOpDesc()->MutableOutputDesc(0), netoutput->GetOpDesc()->MutableInputDesc(0)}};
  EXPECT_EQ(stub_base_pass.update_td_pairs, expected_updated_tensor_desc_pairs);
  EXPECT_EQ(stub_base_pass.GetNodesNeedRePassImmediately(), std::unordered_set<NodePtr>({}));
}

TEST_F(UtestGraphInferBasePassStub, AddCurNodeRepass_NotCallUpdatePeerNode_WhenInferReturnNeedRepass) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  EXPECT_NE(add_node, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.SetInferResult(GRAPH_NODE_NEED_REPASS).Build();

  // do re_pass
  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_infer_times, 1);
  EXPECT_EQ(stub_base_pass.call_update_tensor_desc_times, 0);
//  EXPECT_EQ(stub_base_pass.GetNodesNeedRePassImmediately(), std::unordered_set<NodePtr>({add_node}));
}

TEST_F(UtestGraphInferBasePassStub, NotAddPeerNodeRepass_AfterUpdatePeerNode_WhenUnchanged) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  auto netoutput = test_graph->FindNode("netoutput");
  EXPECT_NE(add_node, nullptr);
  EXPECT_NE(netoutput, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.Build();

  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_update_tensor_desc_times, 1);
  EXPECT_EQ(stub_base_pass.GetNodesNeedRePassImmediately(), std::unordered_set<NodePtr>({}));
  int times = -1;
  EXPECT_TRUE(AttrUtils::GetInt(add_node->GetOpDesc()->GetOutputDescPtr(0), kInferTimes, times));
  EXPECT_EQ(times, 1);
  times = -1;
  EXPECT_TRUE(AttrUtils::GetInt(netoutput->GetOpDesc()->GetInputDescPtr(0), kInferTimes, times));
  EXPECT_EQ(times, 1);
}

TEST_F(UtestGraphInferBasePassStub, AddPeerNodeRepass_AfterUpdatePeerNode_WhenChanged) {
  auto builder = NoSubgraphBuilder();
  auto test_graph = builder.GetGraph();
  auto add_node = test_graph->FindNode("add1");
  auto netoutput = test_graph->FindNode("netoutput");
  EXPECT_NE(add_node, nullptr);
  EXPECT_NE(netoutput, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.SetTdChangedFlag(true).Build();

  EXPECT_EQ(stub_base_pass.Run(add_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_update_tensor_desc_times, 1);
//  EXPECT_EQ(stub_base_pass.GetNodesNeedRePassImmediately(), std::unordered_set<NodePtr>({netoutput}));
}

TEST_F(UtestGraphInferBasePassStub, TestUpdateSubgraphData_WhenBeforeSubgraph) {
  auto builder = RootGraphBuilder();
  auto parent_graph = builder.GetGraph();
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  auto case_node = parent_graph->FindNode("case1");
  auto data1 = subgraphs[0]->FindNode("data1_1");
  auto data2 = subgraphs[0]->FindNode("data2_1");
  EXPECT_NE(case_node, nullptr);
  EXPECT_NE(data1, nullptr);
  EXPECT_NE(data2, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.SetInferResult(GRAPH_NODE_NEED_REPASS).Build();

  EXPECT_EQ(stub_base_pass.Run(case_node), SUCCESS);
  // when GRAPH_NODE_NEED_REPASS, not update peer node, only update two data, update input and output, 2*2
  EXPECT_EQ(stub_base_pass.call_update_tensor_desc_times, 4);
  std::vector<std::pair<GeTensorDescPtr, GeTensorDescPtr>> expected_updated_tensor_desc_pairs = {
    {case_node->GetOpDesc()->MutableInputDesc(0), data1->GetOpDesc()->MutableInputDesc(0)},
    {case_node->GetOpDesc()->MutableInputDesc(0), data1->GetOpDesc()->MutableOutputDesc(0)},
    {case_node->GetOpDesc()->MutableInputDesc(1), data2->GetOpDesc()->MutableInputDesc(0)},
    {case_node->GetOpDesc()->MutableInputDesc(1), data2->GetOpDesc()->MutableOutputDesc(0)},
  };
  EXPECT_EQ(stub_base_pass.update_td_pairs, expected_updated_tensor_desc_pairs);
}

TEST_F(UtestGraphInferBasePassStub, TestUpdateParentNodeOutput_WhenAfterSubgraph) {
  auto builder = RootGraphBuilder();
  auto parent_graph = builder.GetGraph();
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.Build();
  stub_base_pass.SetOption(kOptimizeAfterSubGraph, "");

  EXPECT_EQ(stub_base_pass.Run(case_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_update_from_subgraph_times, 1);
  EXPECT_EQ(stub_base_pass.call_update_from_subgraph_multi_dims_times, 0);
}

TEST_F(UtestGraphInferBasePassStub, TestUpdateParentNodeOutputForMultiDims_WhenAfterSubgraph) {
  auto builder = RootGraphBuilder();
  auto parent_graph = builder.GetGraph();
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  auto case_node = parent_graph->FindNode("case1");
  auto set_ret = AttrUtils::SetInt(case_node->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
  EXPECT_EQ(set_ret, true);
  EXPECT_NE(case_node, nullptr);
  ChildPassBuilder pass_builder;
  auto stub_base_pass = pass_builder.Build();
  stub_base_pass.SetOption(kOptimizeAfterSubGraph, "");

  EXPECT_EQ(stub_base_pass.Run(case_node), SUCCESS);
  EXPECT_EQ(stub_base_pass.call_update_from_subgraph_times, 0);
  EXPECT_EQ(stub_base_pass.call_update_from_subgraph_multi_dims_times, 1);
}
}  // namespace ge