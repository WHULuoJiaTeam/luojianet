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
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/host_cpu/host_cpu_node_executor.h"
#include "common/model/ge_root_model.h"
#include "graph/passes/graph_builder_utils.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "graph/manager/graph_mem_manager.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[2];
}__attribute__((packed));
}  // namespace

class UtestHostCpuNodeTask : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHostCpuNodeTask, test_load) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  std::unique_ptr<NodeItem> node_item;
  ASSERT_EQ(NodeItem::Create(node, node_item), SUCCESS);
  hybrid_model.node_items_[node] = std::move(node_item);
  hybrid_model.task_defs_[node] = {};

  NodeTaskPtr task = nullptr;
  HostCpuNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), PARAM_INVALID);

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 2;

  domi::TaskDef task_def;
  task_def.set_type(RT_MODEL_TASK_ALL_KERNEL);
  task_def.mutable_kernel()->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  task_def.mutable_kernel()->set_args_size(args.head.length);
  hybrid_model.task_defs_[node] = {task_def};
  hybrid_model.node_items_[node]->num_inputs = 1;
  hybrid_model.node_items_[node]->num_outputs = 1;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);

  domi::TaskDef &host_task_def = hybrid_model.task_defs_[node][0];
  host_task_def.set_type(RT_MODEL_TASK_KERNEL);
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
  domi::KernelContext *context = host_task_def.mutable_kernel()->mutable_context();
  context->set_kernel_type(8);    // ccKernelType::HOST_CPU
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
  HostCpuEngine::GetInstance().constant_folding_handle_ = (void *)0x01;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);
}

TEST_F(UtestHostCpuNodeTask, test_execute) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  std::unique_ptr<NodeItem> node_item;
  ASSERT_EQ(NodeItem::Create(node, node_item), SUCCESS);
  domi::TaskDef task_def;

  HostAicpuNodeTask task(node_item.get(), task_def);
  std::function<void()> call_back = []{};
  NodeState node_state(*node_item, nullptr);
  TaskContext context(nullptr, &node_state, nullptr);
  ASSERT_EQ(task.ExecuteAsync(context, call_back), INTERNAL_ERROR);

  std::function<uint32_t (void *)> run_cpu_kernel = [](void *){ return 0; };
  task.SetRunKernel(run_cpu_kernel);
  ASSERT_EQ(task.ExecuteAsync(context, call_back), SUCCESS);
}

TEST_F(UtestHostCpuNodeTask, test_update_args) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  std::unique_ptr<NodeItem> node_item;
  ASSERT_EQ(NodeItem::Create(node, node_item), SUCCESS);
  NodeState node_state(*node_item, nullptr);
  TaskContext context(nullptr, &node_state, nullptr);

  auto *in_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(1);
  auto tmp = TensorBuffer::Create(in_addr, 1);
  std::shared_ptr<TensorBuffer> input_buffer(tmp.release());
  TensorValue input_start[1] = {TensorValue(input_buffer)};
  context.inputs_start_ = input_start;

  auto *out_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(1);
  tmp = TensorBuffer::Create(out_addr, 1);
  std::shared_ptr<TensorBuffer> output_buffer(tmp.release());
  TensorValue output_start[1] = {TensorValue(output_buffer)};
  context.outputs_start_ = output_start;

  domi::TaskDef task_def;
  HostAicpuNodeTask task(node_item.get(), task_def);
  ASSERT_EQ(task.UpdateArgs(context), INTERNAL_ERROR);

  task.args_size_ = sizeof(AicpuTaskStruct);
  task.args_.reset(new(std::nothrow) uint8_t[task.args_size_]());
  ASSERT_EQ(task.UpdateArgs(context), SUCCESS);
}
} // namespace ge
