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
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include "graph/manager/host_mem_manager.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "common/omg_util.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestHybridModelBuilder : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { }
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

static NodePtr CreateConstantNode(const ComputeGraphPtr &graph, const string &name, size_t size) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, CONSTANTOP);
  op_desc->AddOutputDesc(GeTensorDesc());
  GeTensorPtr value = std::make_shared<GeTensor>(GeTensorDesc(), size);
  (void)AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, value);

  return graph->AddNode(op_desc);
}

TEST_F(UtestHybridModelBuilder, normal_hybrid_model_build) {
/*******************************************************************************
 *      Exit         Identify
 *        \         /       \.
 *         \       /         \.
 *          Switch           Add
 *         /     |            |
 * Active /      |            |
 *       /       |            |
 *  LoopCond     |            |
 *      \        |            |
 *       \       |            |
 *        \      |            |
 *       Less    |            |
 *          \    |       NextIteration
 *           \   |            |
 *            \  |            |   Active
 *            Merge <---------|
 *              |
 *              |   Active
 *              |
 *            Enter
 ******************************************************************************/
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);

  auto data1 = CreateNode(*graph, "data", DATA, 1, 1);
  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto merge1 = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  less1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch_t = CreateNode(*graph, "switch_t", STREAMSWITCH, 2, 0);
  auto switch_f = CreateNode(*graph, "switch_f", STREAMSWITCH, 2, 0);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 2, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  add1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const1", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const2", CONSTANT, 0, 1);
  auto active1 = CreateNode(*graph, "active1", STREAMACTIVE, 0, 0);
  auto active2 = CreateNode(*graph, "active2", STREAMACTIVE, 0, 0);
  auto active3 = CreateNode(*graph, "active3", STREAMACTIVE, 0, 0);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), enter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(value0->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));

  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), exit1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));

  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), ident1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(enter1->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(next1->GetOutControlAnchor(), active3->GetInControlAnchor());
  SetNextIteration(merge1, next1);  // for relink NextIteration --> StreamMerge

  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_t->GetInControlAnchor());  // Test for not merge.

  GraphUtils::AddEdge(loop1->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_f->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_t->GetInControlAnchor());

  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  AttrUtils::SetBool(enter1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(output1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);

  SetControlFlowGroup(enter1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(merge1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(loop1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active2, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(next1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active3, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(exit1, loop1->GetOpDesc()->GetId());

  // Build -> IndexSpecialNodes --> stream_merge_op_nodes_
  // Build -> LoadGraph -> RelinkNextIteration
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildNodeItem --> NodeItem::SetDataSend
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildControlFlowGroup --> NodeItem::SetCtrlSend
  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("AIcoreEngine", NodeExecutorManager::ExecutorType::AICORE);
  engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);
  engine_mapping.emplace("aicpu_tf_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("aicpu_ascend_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("ops_kernel_info_hccl", NodeExecutorManager::ExecutorType::HCCL);
  engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
  engine_mapping.emplace("DNN_VM_HOST_CPU_OP_STORE", NodeExecutorManager::ExecutorType::HOST_CPU);

  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICORE, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICPU_TF, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HCCL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HOST_CPU, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  const auto control_group_index = loop1->GetOpDesc()->GetId();
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.Build(), SUCCESS);

  const auto TestFrameGroup = [&hybrid_model](const NodePtr &n, int64_t index) {
    const auto it = hybrid_model.node_items_.find(n);
    ASSERT_NE(hybrid_model.node_items_.end(), it);
    ASSERT_EQ(it->second->frame_index_, index);
    ASSERT_EQ(it->second->parent_frame_, -1);
  };
  auto root_graph = hybrid_model.root_graph_;
  auto enter1_node = root_graph->FindNode("enter");
  auto active1_node = root_graph->FindNode("active1");
  auto active2_node = root_graph->FindNode("active2");
  auto active3_node = root_graph->FindNode("active3");
  auto output1_node = root_graph->FindNode("net_output");
  TestFrameGroup(enter1_node, control_group_index);
  TestFrameGroup(active1_node, control_group_index);
  TestFrameGroup(active2_node, control_group_index);
  TestFrameGroup(active3_node, control_group_index);
  TestFrameGroup(output1_node, -1);

  engine_mapping.clear();
  task_executor.clear();
}

TEST_F(UtestHybridModelBuilder, create_called_invalid) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto node = CreateNode(*graph, "node", PARTITIONEDCALL, 1, 1);
  NodeItem node_item(node);

  ASSERT_EQ(hybrid_model_builder.CreateStreamActiveGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateNextIterationGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(node, &node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateSwitchGroup(node, &node_item), INTERNAL_ERROR);

  ASSERT_EQ(hybrid_model_builder.CreateLabelSetGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELSET;
  ASSERT_EQ(hybrid_model_builder.CreateLabelSetGroup(node, &node_item), UNSUPPORTED);

  ASSERT_EQ(hybrid_model_builder.CreateLabelGotoGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELGOTO;
  ASSERT_EQ(hybrid_model_builder.CreateLabelGotoGroup(node, &node_item), UNSUPPORTED);

  ASSERT_EQ(hybrid_model_builder.CreateLabelSwitchGroup(node, &node_item), INTERNAL_ERROR);
  node_item.node_type = LABELSWITCH;
  ASSERT_EQ(hybrid_model_builder.CreateLabelSwitchGroup(node, &node_item), UNSUPPORTED);
}

TEST_F(UtestHybridModelBuilder, stream_switch_n_group) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto switch_n = CreateNode(*graph, "switch_n", STREAMSWITCHN, 1, 0);
  NodeItem node_item(switch_n);

  // no batch_num
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), INTERNAL_ERROR);

  uint32_t batch_num = 0;
  AttrUtils::SetInt(switch_n->GetOpDesc(), ATTR_NAME_BATCH_NUM, batch_num);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), SUCCESS);

  batch_num = 3;
  AttrUtils::SetInt(switch_n->GetOpDesc(), ATTR_NAME_BATCH_NUM, batch_num);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchNGroup(switch_n, &node_item), SUCCESS);
}

TEST_F(UtestHybridModelBuilder, init_constant_op_host_) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto const_1 = CreateConstantNode(graph, "const_1", 0);
  hybrid_model_builder.constant_op_nodes_.emplace(const_1->GetName(), const_1);
  auto const_2 = CreateConstantNode(graph, "const_2", 10);
  hybrid_model_builder.constant_op_nodes_.emplace(const_2->GetName(), const_2);

  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);

  EXPECT_EQ(hybrid_model_builder.InitConstantOps(), SUCCESS);
  EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 2);
}

TEST_F(UtestHybridModelBuilder, init_host_var_with_host_mem) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
HybridModel hybrid_model(ge_root_model);
HybridModelBuilder hybrid_model_builder(hybrid_model);

OpDescPtr op_desc = std::make_shared<OpDesc>("host_params", VARIABLE);
GeTensorDesc tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
TensorUtils::SetSize(tensor_desc, 512);
op_desc->AddOutputDesc(tensor_desc);
auto host_var = graph->AddNode(op_desc);

hybrid_model.host_variable_nodes_.emplace("host_params", host_var);
std::map<std::string, string> options;
options["ge.exec.placement"] = "HOST";
GetThreadLocalContext().SetGraphOption(options);

EXPECT_EQ(hybrid_model_builder.InitVariableTensors(), SUCCESS);
EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 1);
}

TEST_F(UtestHybridModelBuilder, init_host_var_with_host_shared_mem) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
HybridModel hybrid_model(ge_root_model);
HybridModelBuilder hybrid_model_builder(hybrid_model);

OpDescPtr op_desc = std::make_shared<OpDesc>("host_params", VARIABLE);
GeTensorDesc tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
TensorUtils::SetSize(tensor_desc, 512);
op_desc->AddOutputDesc(tensor_desc);
auto host_var = graph->AddNode(op_desc);

hybrid_model.host_variable_nodes_.emplace("host_params", host_var);
std::map<std::string, string> options;
options["ge.exec.placement"] = "HOST";
GetThreadLocalContext().SetGraphOption(options);

SharedMemInfo info;
uint8_t tmp(0);
info.device_address = &tmp;
std::shared_ptr<AlignedPtr> aligned_ptr = std::make_shared<AlignedPtr>(512, 16);
info.host_aligned_ptr = aligned_ptr;
info.fd=0;
info.mem_size = 100;
info.op_name = "host_params";
HostMemManager::Instance().var_memory_base_map_["host_params"] = info;



EXPECT_EQ(hybrid_model_builder.InitVariableTensors(), SUCCESS);
EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 1);
HostMemManager::Instance().var_memory_base_map_.clear();
}

TEST_F(UtestHybridModelBuilder, TestInitHcclExecutorOnDemand) {
  NodeExecutorManager::GetInstance().builders_.erase(NodeExecutorManager::ExecutorType::HCCL);
  // build aicore task
  domi::ModelTaskDef model_task_def;
  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  // No hccl task
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  task_def->set_type(RT_MODEL_TASK_MEMCPY_ASYNC);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);

  // get executor failed due to no builder
  task_def = model_task_def_ptr->add_task();
  task_def->set_type(RT_MODEL_TASK_HCCL);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), INTERNAL_ERROR);

  // get executor success
  REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, NodeExecutor);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);

  // repeat get, do not access builder
  NodeExecutorManager::GetInstance().builders_.erase(NodeExecutorManager::ExecutorType::HCCL);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);
}

TEST_F(UtestHybridModelBuilder, copy_graph_success) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
HybridModel hybrid_model(ge_root_model);
HybridModelBuilder hybrid_model_builder(hybrid_model);

Status st = hybrid_model_builder.CopyGraph();
EXPECT_EQ(st, SUCCESS);
}
} // namespace ge
