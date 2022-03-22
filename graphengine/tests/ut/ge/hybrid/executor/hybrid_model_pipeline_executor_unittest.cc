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

#define private public
#define protected public
#include "hybrid/executor/hybrid_model_pipeline_executor.h"
#include "graph/ge_context.h"

namespace ge {
using namespace hybrid;

class UtestStageExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

TEST_F(UtestStageExecutor, run_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>(graph);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_ = std::unique_ptr<GraphItem>(new(std::nothrow)GraphItem());

  PipeExecutionConfig config;
  config.device_id = 0;
  config.num_executors = 2;
  config.num_stages = 1;
  config.iteration_end = 2;
  rtCtxGetCurrent(&config.rt_context);
  StageExecutor executor(0, &hybrid_model, &config);
  StageExecutor next_executor(1, &hybrid_model, &config);
  executor.SetNext(&next_executor);
  EXPECT_EQ(executor.Init(), SUCCESS);

  auto allocator = NpuMemoryAllocator::GetAllocator(config.device_id);
  EXPECT_NE(allocator, nullptr);
  StageExecutor::StageTask task_info_1;
  task_info_1.stage = 0;
  task_info_1.iteration = 0;
  EXPECT_EQ(rtEventCreate(&task_info_1.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_1), SUCCESS);
  EXPECT_EQ(executor.Start({}, {}, 2), SUCCESS);

  StageExecutor::StageTask task_info_2;
  task_info_2.stage = 0;
  task_info_2.iteration = 1;
  EXPECT_EQ(rtEventCreate(&task_info_2.event), RT_ERROR_NONE);
  EXPECT_EQ(executor.ExecuteAsync(task_info_2), SUCCESS);
  EXPECT_EQ(executor.Start({}, {}, 2), SUCCESS);
  executor.Reset();
}
} // namespace ge
