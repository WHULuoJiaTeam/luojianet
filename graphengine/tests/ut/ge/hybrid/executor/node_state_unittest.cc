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
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/model/graph_item.h"
#include "graph/utils/graph_utils.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestNodeState : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

TEST_F(UtestNodeState, merge_await_shapes_ready) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  const auto data0 = CreateNode(*graph, "data", DATA, 1, 1);
  const auto data1 = CreateNode(*graph, "data1", DATA, 1, 1);
  const auto merge1 = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);
  const auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  GraphItem graph_item;
  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(merge1, node_item);
  NodeState node_state(*node_item, &subgraph_context);

  // Not dynamic.
  ASSERT_EQ(node_state.shape_inference_state_.AwaitShapesReady(graph_context), SUCCESS);

  // Not set merge index.
  node_item->is_dynamic = true;
  ASSERT_EQ(node_state.shape_inference_state_.AwaitShapesReady(graph_context), FAILED);

  // merge index out of bound.
  AttrUtils::SetInt(merge1->GetOpDesc(), ATTR_NAME_MERGE_INPUT_INDEX, 3);
  ASSERT_EQ(node_state.shape_inference_state_.AwaitShapesReady(graph_context), FAILED);

  AttrUtils::SetInt(merge1->GetOpDesc(), ATTR_NAME_MERGE_INPUT_INDEX, 1);
  ASSERT_EQ(node_state.shape_inference_state_.AwaitShapesReady(graph_context), SUCCESS);
}

} // namespace ge