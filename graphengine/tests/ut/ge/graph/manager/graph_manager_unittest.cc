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

#include <memory>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <future>

#define protected public
#define private public
#include "graph/manager/graph_manager.h"
#include "init/gelib.h"

#include "common/math/math_util.h"
#include "common/thread_pool.h"
#include "common/dump/dump_manager.h"
#include "analyzer/analyzer.h"
#include "common/ge_call_wrapper.h"
#include "common/local_context.h"
#include "common/transop_util.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/partition/dynamic_shape_partition.h"
#include "graph/passes/enter_pass.h"
#include "graph/partition/stage_partition.h"
#include "graph/passes/addn_pass.h"
#include "graph/passes/bitcast_pass.h"
#include "graph/passes/assign_remove_pass.h"
#include "graph/passes/inplace_support_check_pass.h"
#include "graph/passes/atomic_addr_clean_pass.h"
#include "graph/passes/attach_stream_label_pass.h"
#include "graph/passes/cast_remove_pass.h"
#include "graph/passes/common_subexpression_elimination_pass.h"
#include "graph/passes/compile_nodes_pass.h"
#include "graph/passes/cond_remove_pass.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/passes/constant_fuse_same_pass.h"
#include "graph/passes/control_trigger_pass.h"
#include "graph/passes/ctrl_edge_transfer_pass.h"
#include "graph/passes/dimension_adjust_pass.h"
#include "graph/passes/dimension_compute_pass.h"
#include "graph/passes/flow_ctrl_pass.h"
#include "graph/passes/fuse_data_nodes_with_common_input_pass.h"
#include "graph/passes/identity_pass.h"
#include "graph/passes/input_output_connection_identify_pass.h"
#include "graph/passes/iterator_op_pass.h"
#include "graph/passes/link_gen_mask_nodes_pass.h"
#include "graph/passes/mark_graph_unknown_status_pass.h"
#include "graph/passes/merge_pass.h"
#include "graph/passes/merge_input_memcpy_pass.h"
#include "graph/passes/merge_to_stream_merge_pass.h"
#include "graph/passes/multi_batch_pass.h"
#include "graph/passes/next_iteration_pass.h"
#include "graph/passes/permute_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph/passes/ref_identity_delete_op_pass.h"
#include "graph/passes/remove_same_const_pass.h"
#include "graph/passes/reshape_recovery_pass.h"
#include "graph/passes/reshape_remove_pass.h"
#include "graph/passes/same_transdata_breadth_fusion_pass.h"
#include "graph/passes/subgraph_pass.h"
#include "graph/passes/switch_data_edges_bypass.h"
#include "graph/passes/switch_dead_branch_elimination.h"
#include "graph/passes/switch_logic_remove_pass.h"
#include "graph/passes/switch_to_stream_switch_pass.h"
#include "graph/passes/transop_breadth_fusion_pass.h"
#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"
#include "graph/passes/transop_symmetry_elimination_pass.h"
#include "graph/passes/transop_without_reshape_fusion_pass.h"
#include "graph/passes/transpose_transdata_pass.h"
#include "graph/passes/useless_control_out_remove_pass.h"
#include "graph/passes/variable_op_pass.h"
#include "graph/passes/variable_ref_delete_op_pass.h"
#include "graph/passes/variable_ref_useless_control_out_delete_pass.h"
#include "graph/passes/end_of_sequence_add_control_pass.h"
#include "graph/passes/subexpression_migration_pass.h"
#include "graph/passes/subgraph_const_migration_pass.h"
#include "graph/passes/unused_args_clean_pass.h"
#include "graph/passes/global_step_insert_pass.h"
#include "graph/passes/memcpy_addr_async_pass.h"
#include "graph/passes/hccl_continuous_memcpy_pass.h"
#include "graph/build/label_allocator.h"
#include "graph/utils/tensor_adapter.h"
#include "inc/pass_manager.h"
#include "ir_build/option_utils.h"
#include "common/local_context.h"
#include "common/omg_util.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "../passes/graph_builder_utils.h"
#include "register/custom_pass_helper.h"
#include "graph/ops_stub.h"
#include "ge_attr_value.h"

using namespace std;
using namespace testing;
using namespace domi;

namespace {
const uint32_t kNotAdded = 0;
const uint32_t kStartAdd = 1;
const uint32_t kDoneAdded = 2;
}

namespace ge {
class UtestGraphManagerTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class StubExecutor : public Executor {
 public:
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
    return SUCCESS;
  }

  Status UnloadGraph(const GeRootModelPtr &ge_root_model, uint32_t graph_id) {
    return SUCCESS;
  }

  Status PushGraph(const RunArgs &args) {
    return SUCCESS;
  }

  Status RunGraph(const GraphNodePtr &graph_node, GraphId graph_id,
                  const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
    return SUCCESS;
  }

  Status RunGraphWithStream(const GraphNodePtr &graph_node, GraphId graph_id, rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs){
    return SUCCESS;
  }
};

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}
/*      Data
 *       |
 *      Relu       Const
 *       |
 *    Netoutput
 */

ge::ComputeGraphPtr CreateGraphWithIsolatedConst() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto relu = builder.AddNode("addn1", "Relu", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);

  builder.AddDataEdge(data, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraphManagerTest, set_and_get_add_graph_flag) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.SetAddGraphCondition(graph_id, 1);
  uint32_t res = graph_manager.GetAddGraphCondition(graph_id);
  EXPECT_EQ(res, 1);
}

TEST_F(UtestGraphManagerTest, test_add_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.SetAddGraphCondition(graph_id, kDoneAdded);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_3) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  Graph graph("test_graph");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  OmgContext context;

  std::future<Status> fut1 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  std::future<Status> fut2 = std::async(std::launch::async,
      &GraphManager::AddGraph, &graph_manager, graph_id, graph, options, context);
  fut1.wait();
  fut2.wait();
  Status status1 = fut1.get();
  Status status2 = fut2.get();
  EXPECT_EQ(status1, ge::SUCCESS);
  EXPECT_EQ(status2, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_4) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  (void)AttrUtils::SetBool(*compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, true);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_5) {
  Graph graph("test_graph");
  auto data = op::Data("Data").set_attr_index(1);
  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());
  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  graph.SetInputs(inputs).SetOutputs(outputs);

  std::map<std::string, std::string> options = {{"ge.exec.dataInputsShapeRange", "0:[-1]"}};
  OmgContext context;
  GraphId graph_id = 1;
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.AddGraph(graph_id, graph, options, context), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  
  // create graph
  Graph graph("test_graph");
  CreateGraph(graph);
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_manager.graph_map_.insert({1, graph_node});

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Status status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::GE_GRAPH_GRAPH_NOT_EXIST);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetRunFlag(true);
  status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_remove_graph_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  StubExecutor stub_executor;
  graph_manager.executor_ = &stub_executor;

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  Graph graph("test_graph");
  CreateGraph(graph);
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_manager.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(graph_manager.RemoveGraph(graph_id), SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread) {
  
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  std::vector<ge::Tensor> input_tensor;
  uint64_t session_id = 0;
  error_message::Context error_context;
  GEThreadLocalContext context;
  RunAsyncCallback callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  bool ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.PreRunThread();
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_pre_run_thread_2) {
  
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;

  GraphId graph_id = 1;
  GraphNodePtr graph_node_1 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_1);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_manager.IncreaseGraphCount(graph_id);
  graph_node_1->SetBuildFlag(true);
  std::vector<ge::Tensor> input_tensor;
  uint64_t session_id = 0;
  error_message::Context error_context;
  GEThreadLocalContext context;
  RunAsyncCallback callback;
  // PreRunArgs args{graph_id, input_tensor, session_id, error_context, context, callback};
  bool ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);
  graph_id = 2;
  GraphNodePtr graph_node_2 = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node_2);
  ret = graph_manager.prerun_args_q_.Push({graph_id, input_tensor, session_id, error_context, context, callback});
  EXPECT_EQ(ret, true);
  graph_manager.PreRunThread();
  // end with failed
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_1) {
  // no need to build
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node, ge_root_model);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_2) {
  // need build while buildflag is true, var format changed
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  arg.callback = [](Status, std::vector<ge::Tensor> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(true);
  graph_node->Lock();
  graph_manager.var_acc_ctrl_.graph_ids_need_rebuild_.insert(graph_id);
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node, ge_root_model);
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_check_incre_build_and_pre_run_3) {
  // need build while buildflag is false, var format unchanged
  GraphId graph_id = 1;
  GraphManager graph_manager;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs arg;
  arg.callback = [](Status, std::vector<ge::Tensor> &) {};
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetBuildFlag(false);
  graph_node->Lock();
  Status status = graph_manager.CheckIncreBuildAndPreRun(arg, graph_node, ge_root_model);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_add_graph_with_copy_fail) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  // create graph
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> options;
  OmgContext context;
  Status status = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(status, ge::SUCCESS);
  status = graph_manager.RemoveGraph(graph_id);
  EXPECT_EQ(status, ge::SUCCESS);
  status = graph_manager.AddGraphWithCopy(graph_id, graph, options, context);
  EXPECT_NE(status, ge::SUCCESS);
}

TEST_F(UtestGraphManagerTest, test_prerunthread_failed_1) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs args;
  error_message::Context error_ctx{1, "1st_stage", "2nd_stage", "log_header"};
  Status st = 0;
  args.callback = [&st](Status st_return, std::vector<ge::Tensor> &) { st = st_return; };
  args.graph_id = graph_id;
  args.session_id = 1;
  args.error_context = error_ctx;
  args.context = GetThreadLocalContext();
  // create graph
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  graph_manager.options_.local_fmk_op_flag = false;
  // need build while buildflag is true, var format changed
  graph_node->SetBuildFlag(true);
  graph_manager.var_acc_ctrl_.graph_ids_need_rebuild_.insert(graph_id);

  graph_manager.graph_map_.insert({graph_id, graph_node});
  graph_manager.graph_count_.insert({graph_id, 1});
  graph_node->SetRunFlag(false);
  // function return.
  graph_manager.prerun_args_q_.Push(args);
  auto t1 = std::thread(&GraphManager::PreRunThread, &graph_manager);
  if (t1.joinable()) {
    t1.join();
  }
  EXPECT_EQ(st, ge::PARAM_INVALID);
}

TEST_F(UtestGraphManagerTest, test_prerunthread_failed_2) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.thread_run_flag_ = true;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(compute_graph);
  GraphManager::PreRunArgs args;
  error_message::Context error_ctx{1, "1st_stage", "2nd_stage", "log_header"};
  Status st;
  args.callback = [&st, &graph_manager](Status st_return, std::vector<ge::Tensor> &) { st = st_return; 
      graph_manager.thread_run_flag_ = false;};
  args.graph_id = graph_id;
  args.session_id = 1;
  args.error_context = error_ctx;
  args.context = GetThreadLocalContext();
  // create graph
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGraph(graph_ptr);

  graph_manager.options_.local_fmk_op_flag = false;
  // need build while buildflag is true, var format changed
  graph_node->SetBuildFlag(true);
  graph_manager.var_acc_ctrl_.graph_ids_need_rebuild_.insert(graph_id);

  graph_manager.graph_map_.insert({graph_id, graph_node});
  graph_manager.graph_count_.insert({graph_id, 1});
  graph_node->SetRunFlag(false);
  // function continue
  int ret = setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", 1);
  EXPECT_EQ(ret, 0);
  graph_manager.prerun_args_q_.Push(args);
  auto t1 = std::thread(&GraphManager::PreRunThread, &graph_manager);
  if (t1.joinable()) {
    t1.join();
  }
  EXPECT_EQ(st, ge::PARAM_INVALID);
}
// TEST_F(UtestGraphManagerTest, ParseInputsDimsForGetNexNosinkAndData_success) {
//   GraphManager graph_manager;

//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

//   // save1
//   ge::OpDescPtr save_op = std::make_shared<ge::OpDesc>();
//   save_op->SetType("Save");
//   save_op->SetName("Save1");
//   save_op->AddInputDesc(ge::GeTensorDesc());
//   save_op->AddOutputDesc(ge::GeTensorDesc());
//   AttrUtils::SetInt(save_op, ATTR_NAME_INDEX, 1);
//   ge::NodePtr save_node = graph->AddNode(save_op);
 
//   std::vector<NodePtr> nodes;
//   nodes.emplace_back(save_node);
//   ge::Tensor tensor;
//   std::vector<Tensor> input_tensors;
//   input_tensors.emplace_back(tensor);
//   auto ret = graph_manager.ParseInputsDimsForGetNexNosinkAndData(nodes, input_tensors);
//   EXPECT_EQ(ret, ge::SUCCESS);
// }

TEST_F(UtestGraphManagerTest, ChangeAndDeleteConst_success) {
  GraphId graph_id = 1;
  GraphManager graph_manager;
  graph_manager.options_.train_graph_flag = true;

  auto graph = CreateGraphWithIsolatedConst();
  graph_manager.ChangeConstTypeWhenTraining(graph);
  auto const1 = graph->FindFirstNodeMatchType("Const");
  EXPECT_EQ(const1, nullptr);

  Status status = graph_manager.RemoveIsolatedConstInThisGraph(graph);
  EXPECT_EQ(status, ge::SUCCESS);
  auto all_nodes = graph->GetDirectNode();
  EXPECT_EQ(all_nodes.size(), 3);
}
} // namespace ge
