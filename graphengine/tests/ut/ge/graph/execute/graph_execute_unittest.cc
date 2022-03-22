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

#include "common/profiling/profiling_manager.h"

#define protected public
#define private public
#include "graph/execute/graph_execute.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#undef private
#undef public


#include <pthread.h>
#include <algorithm>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <future>

using namespace std;
using namespace testing;
using namespace ge;
using namespace domi;

namespace ge {
namespace {
const uint32_t kInvalidModelId = UINT32_MAX;
}

class UtestGraphExecuteTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphExecuteTest, get_execute_model_id_invalid) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(graph);
  auto model_id = executor.GetExecuteModelId(ge_root_model);
  EXPECT_EQ(model_id, kInvalidModelId);
}

TEST_F(UtestGraphExecuteTest, get_execute_model_id_1) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(graph);
  auto model_manager = ModelManager::GetInstance();
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model1->SetId(1);
  model_manager->InsertModel(1, davinci_model1);
  ge_root_model->SetModelId(1);
  auto model_id = executor.GetExecuteModelId(ge_root_model);
  EXPECT_EQ(model_id, 1);
}

TEST_F(UtestGraphExecuteTest, get_execute_model_id_2) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(graph);
  auto model_manager = ModelManager::GetInstance();
  // model1 with 2 load
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model1->SetId(1);
  davinci_model1->data_inputer_ = new DataInputer();
  auto data = MakeShared<InputDataWrapper>();
  davinci_model1->data_inputer_->Push(data);
  davinci_model1->data_inputer_->Push(data);
  model_manager->InsertModel(1, davinci_model1);
  // model 2 with 3 load
  shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model2->SetId(2);
  davinci_model2->data_inputer_ = new DataInputer();
  davinci_model2->data_inputer_->Push(data);
  davinci_model2->data_inputer_->Push(data);
  davinci_model2->data_inputer_->Push(data);
  model_manager->InsertModel(2, davinci_model2);
  // model 3 witH 1 load
  shared_ptr<DavinciModel> davinci_model3 = MakeShared<DavinciModel>(1, nullptr);
  davinci_model3->SetId(3);
  davinci_model3->data_inputer_ = new DataInputer();
  davinci_model3->data_inputer_->Push(data);
  model_manager->InsertModel(3, davinci_model3);

  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  ge_root_model->SetModelId(3);

  auto model_id = executor.GetExecuteModelId(ge_root_model);
  // model 3 is picked for having least loads
  EXPECT_EQ(model_id, 3);
}

TEST_F(UtestGraphExecuteTest, test_set_callback) {
  GraphExecutor executor;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test");
  // is_unknown_shape_graph_ = false
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>(graph);
  RunAsyncCallback callback = [](Status, std::vector<ge::Tensor> &) {};

  auto model_manager = ModelManager::GetInstance();
  auto listener = MakeShared<RunAsyncListener>();
  shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
  davinci_model1->SetId(1);
  model_manager->InsertModel(1, davinci_model1);
  auto status = executor.SetCallback(1, ge_root_model, callback);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_without_subscribe) {
  GraphExecutor executor;
  auto ret = executor.ModelSubscribe(1);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_failed1) {
  GraphExecutor executor;
  uint32_t graph_id = 1;
  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.SetSubscribeInfo(0, 1, true);
  auto ret = executor.ModelSubscribe(graph_id);
  profiling_manager.CleanSubscribeInfo();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_failed2) {
  GraphExecutor executor;
  uint32_t graph_id = 1;
  uint32_t model_id = 1;
  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.SetSubscribeInfo(0, 1, true);
  profiling_manager.SetGraphIdToModelMap(2, model_id);
  auto ret = executor.ModelSubscribe(graph_id);
  profiling_manager.CleanSubscribeInfo();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphExecuteTest, test_with_subscribe_success) {
  GraphExecutor executor;
  uint32_t graph_id = 1;
  uint32_t model_id = 1;
  GraphNodePtr graph_node = std::make_shared<GraphNode>(graph_id);
  DavinciModel model(model_id, nullptr);
  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.SetSubscribeInfo(0, 1, true);
  profiling_manager.SetGraphIdToModelMap(graph_id, model_id);
  auto ret = executor.ModelSubscribe(graph_id);
  profiling_manager.CleanSubscribeInfo();
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge