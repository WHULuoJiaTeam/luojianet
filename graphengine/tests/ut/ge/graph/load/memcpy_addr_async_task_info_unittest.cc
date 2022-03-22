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
#include "graph/load/model_manager/task_info/memcpy_addr_async_task_info.h"

namespace ge {
class UtestMemcpyAddrAsyncTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

extern OpDescPtr CreateOpDesc(string name, string type);

TEST_F(UtestMemcpyAddrAsyncTaskInfo, success_memcpy_addr_async_task_init) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;

  // DavinciModel is null
  MemcpyAddrAsyncTaskInfo memcpy_addr_async_task_info;
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, nullptr), PARAM_INVALID);

  // SetStream failed.
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), FAILED);

  // GetOpByIndex src failed
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), INTERNAL_ERROR);

  // GetRuntimeAddress src failed.
  model.op_list_[6] = CreateOpDesc("memcpyaddrasync", MEMCPYADDRASYNC);
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), PARAM_INVALID);

  // GetRuntimeAddress dst failed.
  memcpy_async->set_src(0x08003000);
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), PARAM_INVALID);

  memcpy_async->set_dst(0x08008000);
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, success_memcpy_async_task_init_failed) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  task_def.set_stream_id(0);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;


  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYADDRASYNC);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});

  // DavinciModel is null
  MemcpyAddrAsyncTaskInfo memcpy_addr_async_task_info;
  EXPECT_EQ(memcpy_addr_async_task_info.Init(task_def, &model), PARAM_INVALID);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, success_memcpy_async_calculate_args) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(0x08003000);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(0);

  // DavinciModel is null
  MemcpyAddrAsyncTaskInfo memcpy_addr_async_task_info;
  EXPECT_EQ(memcpy_addr_async_task_info.CalculateArgs(task_def, &model), SUCCESS);
}

}  // namespace ge
