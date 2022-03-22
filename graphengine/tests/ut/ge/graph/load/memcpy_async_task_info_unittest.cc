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
#include "graph/load/model_manager/task_info/memcpy_async_task_info.h"


namespace ge {
class UtestMemcpyAsyncTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  AttrUtils::SetFloat(op_desc, ATTR_NAME_ALPHA, 0);
  AttrUtils::SetFloat(op_desc, ATTR_NAME_BETA, 0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  AttrUtils::SetListStr(op_desc, ATTR_NAME_WEIGHT_NAME, {});
  AttrUtils::SetInt(op_desc, POOLING_ATTR_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_PAD_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_DATA_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_CEIL_MODE, 0);
  AttrUtils::SetInt(op_desc, POOLING_ATTR_NAN_OPT, 0);
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_WINDOW, {});
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_PAD, {});
  AttrUtils::SetListInt(op_desc, POOLING_ATTR_STRIDE, {});
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, {1, 1});
  AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, 0);
  return op_desc;
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_init) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;

  MemcpyAsyncTaskInfo memcpy_async_task_info;

  // GetOpByIndex src failed
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  memcpy_async->set_src(0x08008000);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), PARAM_INVALID);

  // set OpDesc attr
  std::vector<int64_t> memory_type = { RT_MEMORY_TS_4G };
  AttrUtils::SetListInt(model.op_list_[6], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  memcpy_async->set_dst_max(0);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), FAILED);

  memcpy_async->set_dst_max(0);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), FAILED);


  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_init_failed) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;


  // DavinciModel is null
  MemcpyAsyncTaskInfo memcpy_async_task_info;
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr), PARAM_INVALID);

  // SetStream failed
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr), PARAM_INVALID);

  // GetOpByIndex failed
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), PARAM_INVALID);
  memcpy_async->set_src(0x08008000);

  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), PARAM_INVALID);
  memcpy_async->set_dst(0x08003000);

  // set OpDesc attr
  std::vector<int64_t> memory_type = { RT_MEMORY_TS_4G };
  AttrUtils::SetListInt(model.op_list_[6], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type);
  memcpy_async->set_dst_max(0);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr), PARAM_INVALID);
  memcpy_async->set_dst_max(512);


  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), SUCCESS);

  memcpy_async->set_dst(0x08009000);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_distribute) {
  DavinciModel model(0, nullptr);
  model.SetKnownNode(true);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;

  MemcpyAsyncTaskInfo memcpy_async_task_info;

  // GetOpByIndex src failed
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  memcpy_async->set_src(0x08008000);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), PARAM_INVALID);

  // set OpDesc attr
  AttrUtils::SetStr(model.op_list_[6], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  memcpy_async->set_dst_max(0);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), SUCCESS);

  memcpy_async->set_dst_max(0);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model), SUCCESS);


  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_distribute) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();

  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(RT_MODEL_TASK_MEMCPY_ASYNC);
  domi::KernelDef *kernel_def = task_def->mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task_def->type()));

  model.task_list_ = { task_info };
  model.ge_model_->SetModelTaskDef(model_task_def);

  EXPECT_EQ(model.DistributeTask(), SUCCESS);
  EXPECT_EQ(task_info->Distribute(), SUCCESS);
  task_info->Release();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_calculate_args) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(0x08003000);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(0);

  model.op_list_[0] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  AttrUtils::SetStr(model.op_list_[0], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");

  // DavinciModel is null
  MemcpyAsyncTaskInfo memcpy_async_task_info;
  EXPECT_EQ(memcpy_async_task_info.CalculateArgs(task_def, &model), SUCCESS);
}

TEST_F(UtestMemcpyAsyncTaskInfo, memcpy_async_update_args) {
  DavinciModel model(0, nullptr);

  MemcpyAsyncTaskInfo memcpy_async_task_info;
  memcpy_async_task_info.davinci_model_ = &model;

  EXPECT_EQ(memcpy_async_task_info.UpdateArgs(), SUCCESS);
}

}  // namespace ge
