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

#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/hccl_task_info.h"

namespace ge {
class UtestHcclTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


// test success GetTaskID
TEST_F(UtestHcclTaskInfo, success_get_task_id) {
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(RT_MODEL_TASK_KERNEL);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  EXPECT_EQ(task_info->GetTaskID(), 0);

  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.GetTaskID(), 0);
}

// test init EventRecordTaskInfo
TEST_F(UtestHcclTaskInfo, success_create_stream) {
  DavinciModel model(0, nullptr);

  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.CreateStream(3, &model, 0), SUCCESS);
}

// test hccl_Distribute
TEST_F(UtestHcclTaskInfo, success_distribute7) {
  DavinciModel model(0, nullptr);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task7 = model_task_def.add_task();
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));
  Status ret = task_info7->Init(task7[0], &model);
  EXPECT_EQ(FAILED, ret);

  std::vector<TaskInfoPtr> task_list;
  task_list.push_back(task_info7);
  model.task_list_ = task_list;

  EXPECT_EQ(task_info7->Release(), SUCCESS);
}

// test hccl_Distribute
TEST_F(UtestHcclTaskInfo, success_distribute7_with_hccl_type) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };

  domi::TaskDef task_def;
  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.Init(task_def, nullptr), PARAM_INVALID);


  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomBroadcast");
  model.op_list_[0] = std::make_shared<OpDesc>("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(hccl_task_info.Init(task_def, &model), SUCCESS);

  task_def.clear_kernel_hccl();
}

// test hccl_GetPrivateDefByTaskDef
TEST_F(UtestHcclTaskInfo, success_hccl_get_private_def_by_task_def) {
  DavinciModel model(0, nullptr);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task7 = model_task_def.add_task();
  task7->set_type(RT_MODEL_TASK_HCCL);
  // for SetStream
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  // for GetPrivateDefByTaskDef
  task7->set_ops_kernel_store_ptr(10);
  std::string value = "hccl_task";
  task7->set_private_def(value);

  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));
  // for Distribute
  EXPECT_EQ(task_info7->Init(task7[0], &model), PARAM_INVALID);

  EXPECT_EQ(task_info7->Release(), SUCCESS);
}

// test hccl_task_TransToGETaskInfo
TEST_F(UtestHcclTaskInfo, success_hccl_trans_to_ge_task_info) {
  DavinciModel model(0, nullptr);

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task7 = model_task_def.add_task();
  // for type
  task7->set_type(RT_MODEL_TASK_HCCL);
  TaskInfoPtr task_info7 = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task7->type()));

  GETaskInfo ge_task;
  HcclTaskInfo hccl_task_info;
  hccl_task_info.TransToGETaskInfo(ge_task);

  EXPECT_EQ(task_info7->Release(), SUCCESS);
}

}  // namespace ge
