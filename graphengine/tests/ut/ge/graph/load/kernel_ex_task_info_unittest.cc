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

#include "graph/load/model_manager/task_info/kernel_ex_task_info.h"
#include "cce/aicpu_engine_struct.h"
#include "tests/depends/runtime/src/runtime_stub.h"

namespace ge {
extern OpDescPtr CreateOpDesc(string name, string type);

class UtestKernelExTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_init) {
  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(1);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), INTERNAL_ERROR);

  kernel_ex_def->clear_op_index();
  kernel_ex_def->set_op_index(0);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);

  kernel_ex_def->set_task_info("KernelEx");
  kernel_ex_def->set_task_info_size(1);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);


  constexpr uint32_t arg_size = sizeof(STR_FWK_OP_KERNEL);
  string value1(arg_size, 'a');
  kernel_ex_def->set_args_size(arg_size);
  kernel_ex_def->set_args(value1);
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);


  task_def.clear_kernel_ex();
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_release) {
  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  kernel_ex_task_info.kernel_buf_ = nullptr;
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  kernel_ex_task_info.input_output_addr_ = nullptr;
  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);

  rtMalloc(&kernel_ex_task_info.kernel_buf_, 64, RT_MEMORY_HBM);
  rtMalloc(&kernel_ex_task_info.input_output_addr_, 64, RT_MEMORY_HBM);
  EXPECT_EQ(kernel_ex_task_info.Release(), SUCCESS);
}

// test kernel_ex_task_Release
TEST_F(UtestKernelExTaskInfo, success_kernel_ex_task_info_copy) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = new uint8_t[model.runtime_param_.mem_size];

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_.push_back(stream);

  domi::TaskDef task_def;
  KernelExTaskInfo kernel_ex_task_info;

  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_task_info_size(150);
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace empty.

  model.op_list_[0]->SetWorkspace({1008});   // offset
  model.op_list_[0]->SetWorkspaceBytes({0});      // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace addr is null.

  model.op_list_[0]->SetWorkspace({1208});   // offset
  model.op_list_[0]->SetWorkspaceBytes({10});     // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), FAILED);  // workspace addr is small.

  model.op_list_[0]->SetWorkspace({1308});   // offset
  model.op_list_[0]->SetWorkspaceBytes({150});    // length
  EXPECT_EQ(kernel_ex_task_info.Init(task_def, &model), SUCCESS);

  task_def.clear_kernel_ex();
  delete [] model.runtime_param_.mem_base;
  model.runtime_param_.mem_base = nullptr;
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_info_calculate_args) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_op_index(0);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  AttrUtils::SetStr(model.op_list_[0], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.CalculateArgs(task_def, &model), FAILED);
}

TEST_F(UtestKernelExTaskInfo, kernel_ex_task_ext_info) {
  const string ext_info = {1, 1, 1, 1, 0, 0, 0, 0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_update_addr) {
  const string ext_info = {3,0,0,0,4,0,0,0,4,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_1) {
  const string ext_info = {7,0,0,0,4,0,0,0,0,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_2) {
  const string ext_info = {7,0,0,0,4,0,0,0,1,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_3) {
  const string ext_info = {7,0,0,0,4,0,0,0,2,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_success_4) {
  const string ext_info = {7,0,0,0,4,0,0,0,3,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_failed_1) {
  const string ext_info = {7,0,0,0,4,0,0,0,4,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_NE(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, parse_topic_type_failed_2) {
  const string ext_info = {7,0,0,0,2,0,0,0,2,0,0,0};
  const OpDescPtr op_desc = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(op_desc, "_AllShape", true);

  KernelExTaskInfo kernel_ex_task_info;
  EXPECT_NE(kernel_ex_task_info.InitTaskExtInfo(ext_info, op_desc), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::TaskDef task_def;
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
  kernel_ex_task_info.op_desc_ = op_desc;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op_fail_01) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::TaskDef task_def;
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);

  kernel_ex_task_info.is_blocking_aicpu_op_ = true;
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);
}

TEST_F(UtestKernelExTaskInfo, blocking_aicpu_op_fail_02) {
  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::TaskDef task_def;
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.op_desc_ = op_desc;
  DavinciModel davinci_model(0, nullptr);
  kernel_ex_task_info.davinci_model_ = &davinci_model;

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtEventReset, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_ex_task_info.InitTaskExtInfo(kernel_ex_def.kernel_ext_info(), op_desc), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_ex_task_info.Distribute(), SUCCESS);
}

}  // namespace ge
