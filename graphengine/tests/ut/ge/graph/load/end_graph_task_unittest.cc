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

#define private public
#define protected public
#include "graph/load/model_manager/task_info/end_graph_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#undef private
#undef protected

using namespace std;

namespace ge {
class UtestEndGraphTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_EndGraphTaskInfo_failed
TEST_F(UtestEndGraphTask, init_end_graph_task_info) {
  domi::TaskDef task_def;
  EndGraphTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  model.stream_list_.clear();
}
}  // namespace ge
