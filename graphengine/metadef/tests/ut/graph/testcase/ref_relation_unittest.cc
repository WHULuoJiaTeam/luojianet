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

#include "graph/ref_relation.h"
#include "graph/compute_graph.h"
#include "graph/utils/mem_utils.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
class UtestRefRelation : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "", int in_num = 0, int out_num = 0) {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; ++i) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; ++i) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});

  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  bool support_dynamic = true;
  ge::AttrUtils::GetBool(op_desc, "support_dynamicshape", support_dynamic);
  return op_desc;
}

TEST_F(UtestRefRelation, build_ref_relations_fail) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto partitioned_call_op_desc = CreateOpDesc("partitioned_call", PARTITIONEDCALL, 3, 1);
  auto partitioned_call_node = root_graph->AddNode(partitioned_call_op_desc);
  partitioned_call_op_desc->AddSubgraphName("f");
  partitioned_call_op_desc->SetSubgraphInstanceName(0, "sub_graph");

  ComputeGraphPtr sub_sub_graph1 = std::make_shared<ComputeGraph>("while_cond");
  {
    OpDescPtr sub_sub_graph_while_cond_data_op_desc = CreateOpDesc("cond_data", DATA);
    NodePtr sub_sub_graph_while_cond_data_node = sub_sub_graph1->AddNode(sub_sub_graph_while_cond_data_op_desc);
    sub_sub_graph1->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_sub_graph1);
  }

  ComputeGraphPtr sub_sub_graph2 = std::make_shared<ComputeGraph>("while_body");
  {
    OpDescPtr sub_sub_graph_while_body_data_op_desc = CreateOpDesc("body_data", DATA);
    NodePtr sub_sub_graph_while_body_data_node = sub_sub_graph2->AddNode(sub_sub_graph_while_body_data_op_desc);
    sub_sub_graph2->SetGraphUnknownFlag(true);
    sub_sub_graph2->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_sub_graph2);
  }

  // Will unfold to merged_graph.
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  {
    OpDescPtr sub_graph_data1_op_desc = CreateOpDesc("data1", DATA, 1, 1);
    OpDescPtr sub_graph_data2_op_desc = CreateOpDesc("data2", DATA, 1, 1);
    OpDescPtr sub_graph_data3_op_desc = CreateOpDesc("data3", DATA, 1, 1);
    NodePtr sub_graph_data1_node = sub_graph->AddNode(sub_graph_data1_op_desc);
    NodePtr sub_graph_data2_node = sub_graph->AddNode(sub_graph_data2_op_desc);
    NodePtr sub_graph_data3_node = sub_graph->AddNode(sub_graph_data3_op_desc);

    AttrUtils::SetInt(sub_graph_data1_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(sub_graph_data2_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(sub_graph_data3_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 2);

    OpDescPtr sub_graph_while_op_desc = CreateOpDesc("while", WHILE, 2, 2);
    NodePtr sub_graph_while_node = sub_graph->AddNode(sub_graph_while_op_desc);
    sub_sub_graph1->SetParentNode(sub_graph_while_node);
    sub_sub_graph2->SetParentNode(sub_graph_while_node);
    sub_graph_while_op_desc->AddSubgraphName("while_cond");
    sub_graph_while_op_desc->SetSubgraphInstanceName(0, "while_cond");
    sub_graph_while_op_desc->AddSubgraphName("while_body");
    sub_graph_while_op_desc->SetSubgraphInstanceName(1, "while_body");

    OpDescPtr sub_graph_matmul_op_desc = CreateOpDesc("matmul", MATMUL, 2, 1);
    NodePtr sub_graph_matmul_node = sub_graph->AddNode(sub_graph_matmul_op_desc);

    OpDescPtr sub_graph_output_op_desc = CreateOpDesc("output", NETOUTPUT, 1, 1);
    NodePtr sub_graph_output_node = sub_graph->AddNode(sub_graph_output_op_desc);

    GraphUtils::AddEdge(sub_graph_data1_node->GetOutDataAnchor(0), sub_graph_while_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_graph_data2_node->GetOutDataAnchor(0), sub_graph_while_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_graph_data3_node->GetOutDataAnchor(0), sub_graph_matmul_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_graph_while_node->GetOutDataAnchor(0), sub_graph_matmul_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_graph_matmul_node->GetOutDataAnchor(0), sub_graph_output_node->GetInDataAnchor(0));

    sub_graph->SetGraphUnknownFlag(true);
    sub_graph->SetParentNode(partitioned_call_node);
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_graph);
  }

  OpDescPtr graph_data1_op_desc = CreateOpDesc("data1", DATA, 1, 1);
  OpDescPtr graph_data2_op_desc = CreateOpDesc("data2", DATA, 1, 1);
  OpDescPtr graph_data3_op_desc = CreateOpDesc("data3", DATA, 1, 1);
  NodePtr graph_data1_node = root_graph->AddNode(graph_data1_op_desc);
  NodePtr graph_data2_node = root_graph->AddNode(graph_data2_op_desc);
  NodePtr graph_data3_node = root_graph->AddNode(graph_data3_op_desc);
  AttrUtils::SetInt(graph_data1_op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(graph_data2_op_desc, ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(graph_data3_op_desc, ATTR_NAME_INDEX, 2);
  GraphUtils::AddEdge(graph_data1_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(graph_data2_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(graph_data3_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(2));

  RefRelations ref;
  EXPECT_NE(ref.BuildRefRelations(*root_graph.get()), ge::SUCCESS);
}
} // namespace ge
