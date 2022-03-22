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
#include "runtime/rt.h"

#define protected public
#define private public
#include "graph/utils/node_utils.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/debug/log.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "graph/load/model_manager/tbe_handle_store.h"
#include "graph/manager/graph_mem_allocator.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "graph/types.h"
#include "graph/utils/tensor_utils.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "single_op/task/build_task_utils.h"
#include "graph/op_desc_impl.h"

using namespace std;

namespace ge {
using namespace hybrid;

class UtestGeHybrid : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    NpuMemoryAllocator::allocators_.clear();
  }
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

TEST_F(UtestGeHybrid, aicore_op_task_init_success) {
  // build aicore task
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def.mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  ASSERT_EQ(aicore_task->Init(*op_desc.get(), task_def), SUCCESS);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  char *handle = "";
  aicore_task->handle_ = handle;
  aicore_task->tiling_key_ = 1;
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
}

TEST_F(UtestGeHybrid, aicore_op_task_init_success2) {
  // build aicore task
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  aicore_task->is_single_op_ = true;
  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_KERNEL);
  domi::KernelDef *kernel = task_def.mutable_kernel();
  kernel->set_block_dim(32);
  kernel->set_args_size(64);
  string args(64, '1');
  kernel->set_args(args.data(), 64);
  domi::KernelContext *context = kernel->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  ASSERT_EQ(aicore_task->InitWithTaskDef(*op_desc.get(), task_def), SUCCESS);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  char *handle = "";
  aicore_task->handle_ = handle;
  aicore_task->tiling_key_ = 1;
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
}

TEST_F(UtestGeHybrid, task_update_tiling_info) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "json");
  ge::AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  ge::AttrUtils::SetInt(op_desc, "op_para_size", 1);
  auto node = graph->AddNode(op_desc);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  SubgraphContext subgraph_context(nullptr, &execution_context);
  auto node_state = subgraph_context.GetOrCreateNodeState(node_item.get());
  ASSERT_EQ(aicore_task->InitTilingInfo(*op_desc), SUCCESS);
  ASSERT_EQ(aicore_task->UpdateTilingInfo(*node_state->GetTaskContext()), SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_failed) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.Build(), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), INTERNAL_ERROR);
}

TEST_F(UtestGeHybrid, parse_force_infershape_nodes) {
  const char *const kForceInfershape = "_force_infershape_when_running";
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Conv2D", "Conv2D");
  ge::AttrUtils::SetBool(op_desc, kForceInfershape, true);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.ParseForceInfershapeNodes(node, *new_node), SUCCESS);
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
TEST_F(UtestGeHybrid, data_direct_connect) {
  std::unique_ptr<NodeItem> node_item;
  auto root_graph = make_shared<ComputeGraph>("root_graph");
  OpDescPtr op_desc = CreateOpDesc("PartitionedCall", "PartitionedCall");
  auto node = root_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(root_graph);
  auto sub_graph = BuildDataDirectConnectGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("subgraph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  root_graph->AddSubgraph("subgraph", sub_graph);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(root_graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.IdentifyVariableOutputs(*new_node.get(), sub_graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_success) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(RT_MODEL_TASK_ALL_KERNEL);
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(0);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), SUCCESS);
}

TEST_F(UtestGeHybrid, init_weight_success) {
  NpuMemoryAllocator::allocators_.emplace(make_pair(0, nullptr));
  // make graph with sub_graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  OpDescPtr op_desc = CreateOpDesc("if", IF);
  NodePtr node = graph->AddNode(op_desc);
  // make sub graph
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("if_sub_graph");
  OpDescPtr const_op_desc = CreateOpDesc("const", CONSTANT);
  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  (void)TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  ConstGeTensorPtr constTensor_0 =
    std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)&data_vec_0[0], data_vec_0.size() * sizeof(int32_t));
  AttrUtils::SetTensor(const_op_desc, ge::ATTR_NAME_WEIGHTS, constTensor_0);
  const_op_desc->AddOutputDesc(tensor_desc_0);
  NodePtr const_node = sub_graph->AddNode(const_op_desc);
  graph->AddSubgraph("sub", sub_graph);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  //Buffer weight_buffer = Buffer(128,0);
  //ge_sub_model->SetWeight(weight_buffer);
  ge_root_model->SetSubgraphInstanceNameToModel("sub",ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.InitWeights();
  ASSERT_EQ(ret,SUCCESS);
  Buffer weight_buffer = Buffer(128,0);
  ge_sub_model->SetWeight(weight_buffer);
  ret = hybrid_model_builder.InitWeights();
  ASSERT_EQ(ret,PARAM_INVALID);
}

TEST_F(UtestGeHybrid, hybrid_model_executor) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("abc");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(compute_graph);
  HybridModel model(root_model);
  model.root_graph_item_.reset(new GraphItem);
  HybridModel *model_ptr = &model;

  uint32_t device_id = 0;
  rtStream_t stream = nullptr;
  HybridModelExecutor executor(model_ptr, device_id, stream);
  executor.Init();
}

TEST_F(UtestGeHybrid, test_parse_parallel_group) {
  NodeExecutorManager::GetInstance().engine_mapping_.emplace("ops_kernel_info_hccl",
                                                             NodeExecutorManager::ExecutorType::HCCL);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");
  OpDescPtr op_desc = CreateOpDesc("AllReduce", "AllReduce");
  op_desc->SetId(0);
  ge::AttrUtils::SetStr(op_desc, ATTR_NAME_PARALLEL_GROUP, "group_1");
  auto node = compute_graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->node_id = 0;

  op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(compute_graph);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;

  HybridModelBuilder builder(model);
  ASSERT_EQ(builder.CollectParallelGroups(node_item.get()), SUCCESS);

  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 1);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 1);

  OpDescPtr op_desc_1 = CreateOpDesc("subgraph", "PartitionedCall");
  op_desc_1->AddSubgraphName("subgraph");
  auto node_1 = compute_graph->AddNode(op_desc_1);

  ComputeGraphPtr subgraph = MakeShared<ComputeGraph>("subgraph");
  ASSERT_EQ(NodeUtils::SetSubgraph(*node_1, 0, subgraph), GRAPH_SUCCESS);

  std::unique_ptr<NodeItem> node_item_1;
  NodeItem::Create(node_1, node_item_1);
  node_item_1->node_id = 1;

  ASSERT_EQ(builder.CollectParallelGroups(node_item_1.get()), SUCCESS);
  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 1);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 1);

  OpDescPtr op_desc_2 = CreateOpDesc("sub_node_1", "AllReduce");
  ge::AttrUtils::SetStr(op_desc_2, ATTR_NAME_PARALLEL_GROUP, "group_1");
  auto node_2 = subgraph->AddNode(op_desc_2);
  ASSERT_TRUE(node_2 != nullptr);

  OpDescPtr op_desc_3 = CreateOpDesc("sub_node_2", "AllReduce2");
  ge::AttrUtils::SetStr(op_desc_3, ATTR_NAME_PARALLEL_GROUP, "group_2");
  auto node_3 = subgraph->AddNode(op_desc_3);
  ASSERT_TRUE(node_3 != nullptr);

  ASSERT_EQ(builder.CollectParallelGroups(node_item_1.get()), SUCCESS);
  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_["group_1"].size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_["group_2"].size(), 1);

  builder.parallel_group_to_nodes_.clear();
  builder.node_ref_inputs_.clear();
  model.node_items_[node] = std::move(node_item);
  model.node_items_[node_1] = std::move(node_item_1);

  ASSERT_FALSE(model.node_items_[node]->has_observer);
  ASSERT_TRUE(model.node_items_[node_1]->dependents_for_execution.empty());
  ASSERT_EQ(builder.ParseDependentByParallelGroup(), SUCCESS);
  ASSERT_TRUE(model.node_items_[node]->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 1);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution[0], node);

  // repeat parse
  ASSERT_EQ(builder.ParseDependentByParallelGroup(), SUCCESS);
  ASSERT_TRUE(model.node_items_[node]->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 1);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution[0], node);
}

TEST_F(UtestGeHybrid, unfold_subgraphs_success) {
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

  ComputeGraphPtr merged_graph = nullptr;
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(root_graph);
  HybridModel hybrid_model(root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  EXPECT_EQ(hybrid_model_builder.UnfoldSubgraphs(root_graph, merged_graph), SUCCESS);
}

TEST_F(UtestGeHybrid, TestTaskContext) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  GraphItem graph_item;
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  subgraph_context.all_inputs_.resize(2);
  subgraph_context.all_outputs_.resize(1);

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item.get());
  auto task_context = node_state->GetTaskContext();
  ASSERT_TRUE(task_context != nullptr);
  auto desc = task_context->MutableInputDesc(2);
  ASSERT_TRUE(desc == nullptr);
  desc = task_context->MutableOutputDesc(0);
  ASSERT_TRUE(desc != nullptr);
  ASSERT_EQ(desc->GetShape().GetDims(), shape.GetDims());
  GeTensorDesc output_desc;
  ASSERT_EQ(task_context->GetOutputDesc(0, output_desc), SUCCESS);
  ASSERT_EQ(output_desc.GetShape().GetDims(), shape.GetDims());

  desc = task_context->MutableInputDesc(0);
  ASSERT_TRUE(desc != nullptr);
  ASSERT_EQ(desc->GetShape().GetDims(), shape.GetDims());
  GeShape new_shape({8, 2});
  tensor_desc.SetShape(new_shape);
  task_context->UpdateInputDesc(1, tensor_desc);
  GeTensorDesc new_desc;
  ASSERT_EQ(task_context->GetInputDesc(1, new_desc), SUCCESS);
  ASSERT_EQ(new_desc.GetShape().GetDims(), new_shape.GetDims());
}

TEST_F(UtestGeHybrid, hybrid_model_executor_update_args) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());

  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  GraphItem graph_item;
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  subgraph_context.all_inputs_.resize(2);
  subgraph_context.all_outputs_.resize(1);

  auto node_state = subgraph_context.GetOrCreateNodeState(node_item.get());
  auto task_context = node_state->GetTaskContext();

  int32_t buffer[1];
  aicore_task->tiling_buffer_ = TensorBuffer::Create(buffer, sizeof(buffer));
  EXPECT_NE(aicore_task->tiling_buffer_, nullptr);
  aicore_task->max_arg_count_ = 0;
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  aicore_task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2]);
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), SUCCESS);
}

TEST_F(UtestGeHybrid, hybrid_model_executor_check_shape) {
  HybridModelExecutor::ExecuteArgs args;
  GeTensorDescPtr ge_tensor = make_shared<GeTensorDesc>(GeTensorDesc());
  vector<int64_t> dim = {2 , 3};
  ge_tensor->SetShape(GeShape(dim));
  args.input_desc.push_back(ge_tensor);

  // create node
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("God");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDesc tensor_desc(GeShape({2, 3}));
  std::vector<std::pair<int64_t, int64_t>> shape_range({std::pair<int64_t, int64_t>(1, 3),
                                                       std::pair<int64_t, int64_t>(2, 4)});
  tensor_desc.SetShapeRange(shape_range);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  new_node->is_dynamic = true;

  GraphItem graph_item;
  graph_item.input_nodes_.emplace_back(new_node.get());

  Status ret = HybridModelExecutor::CheckInputShapeByShapeRange(&graph_item, args);
  ASSERT_EQ(ret, ge::SUCCESS);

  HybridModelExecutor::ExecuteArgs args1;
  ret = HybridModelExecutor::CheckInputShapeByShapeRange(&graph_item, args1);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);

  HybridModelExecutor::ExecuteArgs args2;
  GeTensorDescPtr ge_tensor2 = make_shared<GeTensorDesc>(GeTensorDesc());
  vector<int64_t> dim2 = {-1 , 3};
  ge_tensor2->SetShape(GeShape(dim2));
  args2.input_desc.push_back(ge_tensor2);

  ret = HybridModelExecutor::CheckInputShapeByShapeRange(&graph_item, args1);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);

  HybridModelExecutor::ExecuteArgs args3;
  ret = HybridModelExecutor::CheckInputShapeByShapeRange(&graph_item, args3);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);
}

TEST_F(UtestGeHybrid, TestOptimizeDependenciesForConstInputs) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(compute_graph);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;
  HybridModelBuilder builder(model);

  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  std::unique_ptr<NodeItem> const_node_item;
  {
    OpDescPtr const_op_desc = CreateOpDesc("Constant", "Const");
    const_op_desc->AddOutputDesc(tensor_desc);
    auto const_node = compute_graph->AddNode(const_op_desc);
    NodeItem::Create(const_node, const_node_item);
  }

  std::unique_ptr<NodeItem> non_const_node_item;
  {
    OpDescPtr op_desc = CreateOpDesc("Add", "Add");
    op_desc->AddOutputDesc(tensor_desc);
    auto const_node = compute_graph->AddNode(op_desc);
    NodeItem::Create(const_node, non_const_node_item);
  }

  std::unique_ptr<NodeItem> known_node_item;
  {
    OpDescPtr known_op_desc = CreateOpDesc("known", "PartitionedCall");
    known_op_desc->AddOutputDesc(tensor_desc);
    known_op_desc->AddOutputDesc(tensor_desc);
    auto known_node = compute_graph->AddNode(known_op_desc);
    NodeItem::Create(known_node, known_node_item);
  }

  std::unique_ptr<NodeItem> dst_node_item;
  {
    OpDescPtr known_op_desc = CreateOpDesc("SomeOp", "SomeOpType ");
    known_op_desc->AddOutputDesc(tensor_desc);
    known_op_desc->AddOutputDesc(tensor_desc);
    auto known_node = compute_graph->AddNode(known_op_desc);
    NodeItem::Create(known_node, dst_node_item);
  }

  float buffer[2 * 16];
  unique_ptr<TensorValue> tensor_value(new TensorValue(buffer, sizeof(buffer)));
  model.constant_tensors_[const_node_item->node] = std::move(tensor_value);

  // Case 1. connect to Const
  auto output_id = 1;
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(output_id, const_node_item.get());
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(0, non_const_node_item.get());
  dst_node_item->dependents_for_shape_inference.emplace_back(const_node_item->node);
  dst_node_item->dependents_for_shape_inference.emplace_back(non_const_node_item->node);

  ASSERT_EQ(builder.OptimizeDependenciesForConstantInputs(), SUCCESS);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference.size(), 1);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference[0], non_const_node_item->node);

  // Case 2. connect to known-subgraph, netoutput connect to Const
  builder.host_input_value_dependencies_.clear();
  dst_node_item->dependents_for_shape_inference.clear();

  builder.known_subgraph_constant_output_refs_[known_node_item.get()].emplace(output_id, const_node_item->node);
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(output_id, known_node_item.get());
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(0, non_const_node_item.get());

  dst_node_item->dependents_for_shape_inference.emplace_back(known_node_item->node);
  dst_node_item->dependents_for_shape_inference.emplace_back(non_const_node_item->node);

  ASSERT_EQ(builder.OptimizeDependenciesForConstantInputs(), SUCCESS);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference.size(), 1);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference[0], non_const_node_item->node);
}

TEST_F(UtestGeHybrid, test_key_for_kernel_bin) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  OpDesc op_desc("Sum", "Sum");
  EXPECT_EQ(aicore_task->GetKeyForTbeKernel(), OP_EXTATTR_NAME_TBE_KERNEL);
  EXPECT_EQ(aicore_task->GetKeyForTvmMagic(), TVM_ATTR_NAME_MAGIC);
  EXPECT_EQ(aicore_task->GetKeyForTvmMetaData(), TVM_ATTR_NAME_METADATA);
  EXPECT_EQ(aicore_task->GetKeyForKernelName(op_desc), "Sum_kernelname");

  auto atomic_task = std::unique_ptr<hybrid::AtomicAddrCleanOpTask>(new(std::nothrow)hybrid::AtomicAddrCleanOpTask());
  EXPECT_EQ(atomic_task->GetKeyForTbeKernel(), EXT_ATTR_ATOMIC_TBE_KERNEL);
  EXPECT_EQ(atomic_task->GetKeyForTvmMagic(), ATOMIC_ATTR_TVM_MAGIC);
  EXPECT_EQ(atomic_task->GetKeyForTvmMetaData(), ATOMIC_ATTR_TVM_METADATA);
  EXPECT_EQ(atomic_task->GetKeyForKernelName(op_desc), "Sum_atomic_kernelname");
}

TEST_F(UtestGeHybrid, test_op_type) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  aicore_task->op_type_ = "Add";
  EXPECT_EQ(aicore_task->GetOpType(), "Add");

  auto atomic_task = std::unique_ptr<hybrid::AtomicAddrCleanOpTask>(new(std::nothrow)hybrid::AtomicAddrCleanOpTask());
  EXPECT_EQ(atomic_task->GetOpType(), "DynamicAtomicAddrClean");
}

TEST_F(UtestGeHybrid, TestParseDependentInputNodesForHccl) {
  NodeExecutorManager::GetInstance().engine_mapping_.emplace("ops_kernel_info_hccl",
                                                             NodeExecutorManager::ExecutorType::HCCL);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  auto node = compute_graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->node_id = 0;

  OpDescPtr op_desc_1 = CreateOpDesc("AllReduce", "AllReduce");
  op_desc_1->SetOpKernelLibName("ops_kernel_info_hccl");
  auto node_1 = compute_graph->AddNode(op_desc_1);
  std::unique_ptr<NodeItem> node_item_1;
  NodeItem::Create(node_1, node_item_1);
  node_item_1->node_id = 1;
  node->GetOutControlAnchor()->LinkTo(node_1->GetInControlAnchor());

  OpDescPtr op_desc_2 = CreateOpDesc("net_output", NETOUTPUT);
  auto node_2 = compute_graph->AddNode(op_desc_2);
  std::unique_ptr<NodeItem> node_item_2;
  NodeItem::Create(node_2, node_item_2);
  node_item_2->node_id = 2;
  node_1->GetOutControlAnchor()->LinkTo(node_2->GetInControlAnchor());

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(compute_graph);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;
  model.node_items_.emplace(node, std::move(node_item));
  model.node_items_.emplace(node_1, std::move(node_item_1));
  model.node_items_.emplace(node_2, std::move(node_item_2));

  HybridModelBuilder builder(model);
  std::vector<std::string> deps;
  ASSERT_EQ(builder.ParseDependentInputNodes(*model.node_items_[node_1], deps), SUCCESS);
  ASSERT_EQ(builder.ParseDependentInputNodes(*model.node_items_[node_2], deps), SUCCESS);
  ASSERT_FALSE(model.GetNodeItem(node)->has_observer);
  ASSERT_TRUE(model.GetNodeItem(node_1)->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 0);
  ASSERT_EQ(model.node_items_[node_2]->dependents_for_execution.size(), 1);
}

TEST_F(UtestGeHybrid, TestParseDependencies) {
  // make graph
  ut::GraphBuilder graph_builder = ut::GraphBuilder("graph");
  auto data = graph_builder.AddNode("Data", "Data", 0, 1);
  auto netoutput = graph_builder.AddNode("Netoutput", "NetOutput", 1, 0);
  graph_builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = graph_builder.GetGraph();

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>(graph);
  HybridModel model(root_model);
  HybridModelBuilder builder(model);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(netoutput, node_item);
  std::unique_ptr<NodeItem> node_item2;
  NodeItem::Create(data, node_item2);
  model.node_items_.emplace(data, std::move(node_item2));

  std::vector<std::string> deps;
  deps.push_back("Data");
  auto op_desc = netoutput->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  auto data_desc = data->GetOpDesc();
  auto tensor = std::make_shared<GeTensor>();
  auto tensor_desc = data_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", tensor);
  std::set<NodePtr> dependent_for_shape_inference;
  ASSERT_EQ(builder.ParseDependencies(*node_item, deps, dependent_for_shape_inference), SUCCESS);
}

TEST_F(UtestGeHybrid, TestTaskExecuteAsync) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphExecutionContext execution_context;
  GraphItem graph_item;
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  subgraph_context.all_inputs_.resize(2);
  subgraph_context.all_outputs_.resize(1);
  auto node_state = subgraph_context.GetOrCreateNodeState(node_item.get());
  auto task_context = *node_state->GetTaskContext();
  ASSERT_NE(BuildTaskUtils::GetTaskInfo(task_context), "");
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  AiCoreNodeTask node_task(std::move(tasks));
  ASSERT_EQ(node_task.ExecuteAsync(task_context, nullptr), SUCCESS);
}
} // namespace ge