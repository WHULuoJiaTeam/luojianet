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
#include <cstdint>
#include <string>

#define private public
#include "inc/graph/ge_local_context.h"
#include "inc/external/ge/ge_api_types.h"
#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#include "utils/graph_utils.h"
#include "graph/passes/parallel_group_pass.h"
#undef private

namespace ge {
namespace {

class UtestGraphPassesParallelGgroupPass : public testing::Test {
 protected:
  UtestGraphPassesParallelGgroupPass() {
    graph_ = std::make_shared<ComputeGraph>("test");
    sub_graph_ = std::make_shared<ComputeGraph>("test_subgraph");
    vector<int64_t> shape_vec{1, 1, 1, 1};
    GeShape shape = GeShape(shape_vec);
    default_tensor_desc_ = std::make_shared<GeTensorDesc>();
    default_tensor_desc_->SetShape(shape);
    default_tensor_desc_->SetFormat(FORMAT_NCHW);
    default_tensor_desc_->SetDataType(DT_FLOAT);
  }

  NodePtr NewNode(const std::string &name, const std::string &type,
                  int input_cnt, int output_cnt, bool isSubgraph = false) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(default_tensor_desc_->Clone());
    }

    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(default_tensor_desc_->Clone());
    }
    NodePtr node = nullptr;
    if (isSubgraph) {
      node = sub_graph_->AddNode(op_desc);
      (void)node->SetOwnerComputeGraph(sub_graph_);
    } else {
      node = graph_->AddNode(op_desc);
      (void)node->SetOwnerComputeGraph(graph_);
    }

    return node;
  }

  void BuildDefaultGraph() {
    ///          input
    ///            \.
    ///           sqrt pred
    ///              \  /
    ///              cast
    ///              /  \.
    ///       switch_t  switch_f
    ///              |  |
    ///              F  T
    ///              |  |
    ///              Merge
    ///               |
    ///              relu
    ///               |
    ///              sqrt1
    input_node_ = NewNode("input", RELU, 0, 1);
    sqrt_node_ = NewNode("sqrt", SQRT, 1, 1);
    pred_node_ = NewNode("pred", GREATER, 2, 1);
    cast_node_ = NewNode("cast", CAST, 2, 2);
    AttrUtils::SetStr(input_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");

    switch_node_t = NewNode("switch_t", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_t->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    switch_node_f = NewNode("switch_f", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_f->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, false);
    output_false_node_ = NewNode("false_output", RELU, 1, 1);
    AttrUtils::SetStr(output_false_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    output_true_node_ = NewNode("true_output", RELU, 1, 1);
    AttrUtils::SetStr(output_true_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    merge_node_ = NewNode("merge", STREAMMERGE, 2, 1);
    relu_node_ = NewNode("relu", RELU, 1, 1);
    sqrt_node1_ = NewNode("sqrt1", SQRT, 1, 1);
    AttrUtils::SetStr(sqrt_node1_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");

    GraphUtils::AddEdge(input_node_->GetOutDataAnchor(0), sqrt_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(0), switch_node_t->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(1), switch_node_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_f->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_t->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));

    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(merge_node_->GetOutDataAnchor(0), relu_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node_->GetOutDataAnchor(0), sqrt_node1_->GetInDataAnchor(0));

    output_false_node_->GetOpDesc()->SetIsInputConst({false});
    output_true_node_->GetOpDesc()->SetIsInputConst({false});
  }

  void BuildDefaultGraph1() {
    ///          input
    ///            \.
    ///           sqrt  pred
    ///              \  /
    ///             Switch
    ///              |  |
    ///          ----F  T----
    ///          \   | /     \.
    ///           \  Merge1 Merge2
    ///            \_________|
    input_node_ = NewNode("input", RELU, 0, 1);
    AttrUtils::SetStr(input_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    pred_node_ = NewNode("pred", GREATER, 2, 1);
    sqrt_node_ = NewNode("sqrt", SQRT, 1, 1);
    cast_node_ = NewNode("cast", CAST, 2, 2);

    switch_node_t = NewNode("switch_t", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_t->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    switch_node_f = NewNode("switch_f", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_f->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, false);
    output_false_node_ = NewNode("false_output", RELU, 1, 2);
    AttrUtils::SetStr(output_false_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    output_true_node_ = NewNode("true_output", RELU, 1, 2);
    AttrUtils::SetStr(output_true_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    merge_node_ = NewNode("merge", STREAMMERGE, 2, 1);
    merge_node1_ = NewNode("merge1", STREAMMERGE, 2, 1);

    GraphUtils::AddEdge(input_node_->GetOutDataAnchor(0), sqrt_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(0), switch_node_t->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(1), switch_node_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_f->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_t->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));

    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(1), merge_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(1), merge_node1_->GetInDataAnchor(1));

    output_false_node_->GetOpDesc()->SetIsInputConst({false});
    output_true_node_->GetOpDesc()->SetIsInputConst({false});
  }


  void BuildDefaultGraph2() {
    ///          input      input1
    ///            \           \.
    ///           sqrt  pred sqrt1  pred1
    ///              \  /       \  /
    ///             Switch     Switch1
    ///              |  |  _______|
    ///              |  | /
    ///          ____F  T____
    ///          \   | /     \.
    ///           \  Merge1 Merge2
    ///            \__________|
    input_node_ = NewNode("input", RELU, 0, 2);
    input_node1_ = NewNode("input_1", RELU, 0, 2);
    sqrt_node_ = NewNode("sqrt", SQRT, 1, 1);
    pred_node_ = NewNode("pred", GREATER, 2, 1);
    sqrt_node1_ = NewNode("sqrt_1", SQRT, 1, 1);
    pred_node1_ = NewNode("pred_1", LESS, 2, 1);
    cast_node_ = NewNode("cast", CAST, 2, 2);
    cast_node1_ = NewNode("cast_1", CAST, 2, 2);
    AttrUtils::SetStr(input_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    AttrUtils::SetStr(input_node1_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "2");

    switch_node_t = NewNode("switch_t", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_t->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    switch_node_f = NewNode("switch_f", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_f->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, false);
    switch_node1_t = NewNode("switch1_t", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node1_t->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    switch_node1_f = NewNode("switch1_f", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node1_f->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, false);
    output_false_node_ = NewNode("false_output", RELU, 2, 2);
    AttrUtils::SetStr(output_false_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    output_true_node_ = NewNode("true_output", RELU, 2, 2);
    AttrUtils::SetStr(output_true_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "2");
    merge_node_ = NewNode("merge", STREAMMERGE, 2, 1);
    merge_node1_ = NewNode("merge1", STREAMMERGE, 2, 1);

    GraphUtils::AddEdge(input_node_->GetOutDataAnchor(0), sqrt_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(0), switch_node_t->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(1), switch_node_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_f->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_t->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));

    GraphUtils::AddEdge(input_node1_->GetOutDataAnchor(0), sqrt_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node1_->GetOutDataAnchor(0), cast_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node1_->GetOutDataAnchor(0), cast_node1_->GetInDataAnchor(1));
    GraphUtils::AddEdge(cast_node1_->GetOutDataAnchor(0), switch_node1_t->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node1_->GetOutDataAnchor(1), switch_node1_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node1_f->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(switch_node1_t->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(1));

    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(1), merge_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(1), merge_node1_->GetInDataAnchor(1));

    output_false_node_->GetOpDesc()->SetIsInputConst({false});
    output_true_node_->GetOpDesc()->SetIsInputConst({false});
  }

  void BuildDefaultGraph3() {
    ///          input
    ///            \
    ///           sqrt  pred
    ///              \  /
    ///             Switch
    ///              |  |
    ///              F  T ------
    ///             / \_/_      \
    ///            /   /  \      \
    ///          Merge    sqrt2  sqrt3
    ///           /         \      \
    ///         sqrt1        \     relu
    ///                       \     \
    ///                        \     sqrt4
    ///                         \     /
    ///                          Merge1
    input_node_ = NewNode("input", RELU, 0, 1);
    AttrUtils::SetStr(input_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    pred_node_ = NewNode("pred", GREATER, 2, 1);
    sqrt_node_ = NewNode("sqrt", SQRT, 1, 1);
    cast_node_ = NewNode("cast", CAST, 2, 2);

    switch_node_t = NewNode("switch_t", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_t->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, true);
    switch_node_f = NewNode("switch_f", STREAMSWITCH, 1, 1);
    AttrUtils::SetBool(switch_node_f->GetOpDesc(), ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG, false);
    output_false_node_ = NewNode("false_output", RELU, 1, 2);
    AttrUtils::SetStr(output_false_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    output_true_node_ = NewNode("true_output", RELU, 1, 2);
    AttrUtils::SetStr(output_true_node_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    merge_node_ = NewNode("merge", STREAMMERGE, 2, 1);
    sqrt_node1_ = NewNode("sqrt1", SQRT, 1, 1);
    AttrUtils::SetStr(sqrt_node1_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    sqrt_node2_ = NewNode("sqrt2", SQRT, 1, 1);
    AttrUtils::SetStr(sqrt_node2_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    sqrt_node3_ = NewNode("sqrt3", SQRT, 1, 1);
    relu_node_ = NewNode("relu", RELU, 1, 1);
    sqrt_node4_ = NewNode("sqrt4", SQRT, 1, 1);
    AttrUtils::SetStr(sqrt_node4_->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
    merge_node1_ = NewNode("merge1", STREAMMERGE, 2, 1);

    GraphUtils::AddEdge(input_node_->GetOutDataAnchor(0), sqrt_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(pred_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node_->GetOutDataAnchor(0), cast_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(0), switch_node_t->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast_node_->GetOutDataAnchor(1), switch_node_f->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_f->GetOutDataAnchor(0), output_false_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(switch_node_t->GetOutDataAnchor(0), output_true_node_->GetInDataAnchor(0));

    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(0), merge_node_->GetInDataAnchor(1));
    GraphUtils::AddEdge(output_false_node_->GetOutDataAnchor(1), sqrt_node2_->GetInDataAnchor(0));
    GraphUtils::AddEdge(output_true_node_->GetOutDataAnchor(1), sqrt_node3_->GetInDataAnchor(0));

    GraphUtils::AddEdge(merge_node_->GetOutDataAnchor(0), sqrt_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node3_->GetOutDataAnchor(0), relu_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node_->GetOutDataAnchor(0), sqrt_node4_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node2_->GetOutDataAnchor(0), merge_node1_->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node4_->GetOutDataAnchor(0), merge_node1_->GetInDataAnchor(1));
    output_false_node_->GetOpDesc()->SetIsInputConst({false});
    output_true_node_->GetOpDesc()->SetIsInputConst({false});
  }

  ComputeGraphPtr graph_;
  ComputeGraphPtr sub_graph_;
  GeTensorDescPtr default_tensor_desc_;
  ParallelGroupPass pass_;
  NodePtr pred_node_;
  NodePtr pred_node1_;
  NodePtr cast_node_;
  NodePtr cast_node1_;
  NodePtr sqrt_node_;
  NodePtr sqrt_node1_;
  NodePtr sqrt_node2_;
  NodePtr sqrt_node3_;
  NodePtr sqrt_node4_;
  NodePtr input_node_;
  NodePtr input_node1_;
  NodePtr switch_node_t;
  NodePtr switch_node_f;
  NodePtr switch_node1_t;
  NodePtr switch_node1_f;
  NodePtr output_false_node_;
  NodePtr output_true_node_;
  NodePtr merge_node_;
  NodePtr merge_node1_;
  NodePtr relu_node_;
};

TEST_F(UtestGraphPassesParallelGgroupPass, null_graph) {
  ComputeGraphPtr graph = nullptr;
  auto ret = pass_.Run(graph);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesParallelGgroupPass, normal_graph) {
  BuildDefaultGraph();
  auto ret = pass_.Run(graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(true, input_node_->GetOutControlAnchor()->IsLinkedWith(cast_node_->GetInControlAnchor()));
  EXPECT_EQ(true, merge_node_->GetOutControlAnchor()->IsLinkedWith(sqrt_node1_->GetInControlAnchor()));
  EXPECT_EQ(false, output_false_node_->GetOutControlAnchor()->IsLinkedWith(output_true_node_->GetInControlAnchor()));
}

TEST_F(UtestGraphPassesParallelGgroupPass, normal_graph1) {
  BuildDefaultGraph1();
  auto ret = pass_.Run(graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(true, input_node_->GetOutControlAnchor()->IsLinkedWith(cast_node_->GetInControlAnchor()));
}

TEST_F(UtestGraphPassesParallelGgroupPass, normal_graph2) {
  BuildDefaultGraph2();
  auto ret = pass_.Run(graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(true, input_node_->GetOutControlAnchor()->IsLinkedWith(cast_node_->GetInControlAnchor()));
  EXPECT_EQ(true, input_node1_->GetOutControlAnchor()->IsLinkedWith(cast_node1_->GetInControlAnchor()));
}

TEST_F(UtestGraphPassesParallelGgroupPass, normal_graph3) {
  std::map<std::string, std::string> options;
  options.emplace(OPTION_GRAPH_RUN_MODE, "1");
  GetThreadLocalContext().SetGraphOption(options);
  BuildDefaultGraph3();
  auto ret = pass_.Run(graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(true, merge_node1_->GetOutControlAnchor()->IsLinkedWith(sqrt_node1_->GetInControlAnchor()));
}

TEST_F(UtestGraphPassesParallelGgroupPass, normal_subgraph) {
  BuildDefaultGraph1();
  NodePtr input_node1 = NewNode("input1", RELU, 0, 1, true);
  NodePtr input_node2 = NewNode("input2", RELU, 0, 1, true);
  NodePtr add = NewNode("add", ADD, 2, 1, true);
  AttrUtils::SetStr(input_node1->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");
  AttrUtils::SetStr(input_node2->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, "1");

  sub_graph_->SetParentNode(input_node_);
  sub_graph_->SetParentGraph(graph_);
  auto ret = graph_->AddSubgraph(sub_graph_->GetName(), sub_graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = input_node_->GetOpDesc()->AddSubgraphName(sub_graph_->GetName());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = input_node_->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph_->GetName());
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = pass_.Run(sub_graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = pass_.Run(graph_);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

}  // namespace
}  // namespace ge
