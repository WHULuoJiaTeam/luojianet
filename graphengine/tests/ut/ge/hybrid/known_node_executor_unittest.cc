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
#include <memory>

#define protected public
#define private public
#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "common/dump/dump_manager.h"
#undef private
#undef protected
#include "graph/manager/graph_mem_allocator.h"
#include "../graph/passes/graph_builder_utils.h"
#include "../inc/graph/utils/graph_utils.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace hybrid;

class UnknownNodeExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
class KnownNodeTaskMock : public KnownNodeTask {
 public:
  KnownNodeTaskMock(std::shared_ptr<DavinciModel> davinci_model): KnownNodeTask(davinci_model) {};
  ~KnownNodeTaskMock() override = default;
  MOCK_METHOD2(DoInitDavinciModel, Status(void *, size_t));
};
}

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  ;
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  bool support_dynamic = true;
  ge::AttrUtils::GetBool(op_desc, "support_dynamicshape", support_dynamic);
  return op_desc;
}

static ComputeGraphPtr BuildDataDirectConnectGraph() {
  const char *kRefIndex = "_parent_node_index";
  ge::ut::GraphBuilder builder("subgraph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 1);
  (void)AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), kRefIndex, 0);

  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UnknownNodeExecutorTest, test_init_davinci_model) {
  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->SetDeviceId(0);
  davinci_model->SetKnownNode(true);

  auto ge_model = make_shared<GeModel>();
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  davinci_model->Assign(ge_model);

  HybridModel model(nullptr);
  KnownNodeTaskMock mock(davinci_model);
  DumpProperties dump_properties;
  dump_properties.enable_dump_ = "1";
  DumpManager::GetInstance().AddDumpProperties(model.GetSessionId(), dump_properties);
  EXPECT_CALL(mock, DoInitDavinciModel).WillRepeatedly(::testing::Return(SUCCESS));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);

  int32_t buffer[8];
  model.weight_buffer_map_.emplace("subgraph", TensorBuffer::Create(buffer, sizeof(buffer)));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);
}

TEST_F(UnknownNodeExecutorTest, TestParseAttrForAllocatingOutputs) {
  ut::GraphBuilder builder("test-graph");
  auto data_node = builder.AddNode("Data0", DATA, 1, 1);
  auto netoutput_node = builder.AddNode("NodeOutput", NETOUTPUT, 2, 2);
  builder.AddDataEdge(data_node, 0, netoutput_node, 0);
  auto const_node = builder.AddNode("Const0", CONSTANT, 0, 1);
  builder.AddDataEdge(const_node, 0, netoutput_node, 1);
  auto graph = builder.GetGraph();

  ut::GraphBuilder builder2("root-graph");
  auto partitioned_call = builder2.AddNode("Node0", PARTITIONEDCALL, 1, 2);
  NodeItem node_item(partitioned_call);
  ASSERT_EQ(KnownNodeExecutor::ParseAttrForAllocatingOutputs(node_item, *graph), SUCCESS);
  ASSERT_EQ(node_item.ref_outputs.size(), 1);
  ASSERT_EQ(node_item.ref_outputs[1], const_node);
  ASSERT_EQ(node_item.reuse_inputs.size(), 1);
  ASSERT_EQ(node_item.reuse_inputs[0], 0);
}

TEST_F(UnknownNodeExecutorTest, TestSetGlobalStep) {
  OpDescPtr op_desc = CreateOpDesc("PartitionedCall", "PartitionedCall");
  auto root_graph = make_shared<ComputeGraph>("root_graph");
  auto node = root_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(root_graph);
  auto sub_graph = BuildDataDirectConnectGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("subgraph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  root_graph->AddSubgraph("subgraph", sub_graph);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(root_graph);
  HybridModel hybrid_model(ge_root_model);
  auto *step_id = new int64_t[1];
  step_id[0] = 520;
  std::unique_ptr<TensorBuffer> tensor_buf;
  tensor_buf = tensor_buf->Create((void *)step_id, sizeof(int64_t));
  hybrid_model.global_step_ = std::move(tensor_buf);
  KnownNodeExecutor known_node_executor;
  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  known_node_executor.SetDaviciModel(hybrid_model, node, davinci_model);
  EXPECT_EQ(*(static_cast<int64_t*>(davinci_model->global_step_addr_)), 520);
}
