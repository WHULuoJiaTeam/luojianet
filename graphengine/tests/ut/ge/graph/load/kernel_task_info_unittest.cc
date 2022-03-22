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
#include "graph/load/model_manager/task_info/kernel_task_info.h"
#include "graph/load/model_manager/task_info/hccl_task_info.h"
#include "tests/depends/runtime/src/runtime_stub.h"

namespace ge {
extern OpDescPtr CreateOpDesc(string name, string type);

class UtestKernelTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

// test KernelTaskInfo Init.
TEST_F(UtestKernelTaskInfo, success_kernel_taskInfo_not_te) {
  DavinciModel model(0, nullptr);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(RT_MODEL_TASK_KERNEL);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  task->stream_id_ = 0;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };

  domi::KernelDef *kernel_def = task->mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  model.op_list_[0] = CreateOpDesc("relu", RELU);
  ctx->set_op_index(0);

  EXPECT_EQ(task_info->Init(*task, &model), FAILED);

  kernel_def->set_block_dim(10);
  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);

  ctx->set_kernel_type(0);
  EXPECT_EQ(task_info->Init(*task, &model), INTERNAL_ERROR);

  task_info->Release();
}

TEST_F(UtestKernelTaskInfo, success_init_kernel_task_info_fail) {
  DavinciModel model(0, nullptr);
  KernelTaskInfo kernel_task_info;
  domi::TaskDef task_def;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();

  model.op_list_[0] = CreateOpDesc("relu", RELU);
  ctx->set_op_index(0);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };

  // Failed by rtGetFunctionByName.
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);
}

// test InitTVMTask failed
TEST_F(UtestKernelTaskInfo, init_tvm_task_fail) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;

  EXPECT_EQ(kernel_task_info.InitTVMTask(0, *kernel_def), PARAM_INVALID);
  task_def.clear_kernel();
}

// test InitTVMTask with kernel_type is TE
TEST_F(UtestKernelTaskInfo, init_tvm_task_info_with_te_kernel_type) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  // DavinciModel is nullptr
  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.Init(task_def, nullptr), PARAM_INVALID);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;

  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));
  kernel_def->set_flowtable("fl", 2);
  kernel_def->set_block_dim(10);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(2);
  ctx->set_op_index(4);
  ctx->set_args_offset("\0\0"); // args_offset = 0
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), PARAM_INVALID);

  ctx->clear_args_offset();
  ctx->set_args_offset("args111111", 10);
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), PARAM_INVALID);

  ctx->clear_op_index();
  ctx->set_op_index(0);
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);


  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask with kernel_type is CUSTOMIZED
TEST_F(UtestKernelTaskInfo, init_kernel_task_info_with_customized_kernel_type) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;

  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));
  kernel_def->set_flowtable("fl", 2);
  kernel_def->set_block_dim(10);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(3);
  ctx->set_op_index(4);
  ctx->set_args_offset("\0\0"); // args_offset = 0
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), PARAM_INVALID);

  ctx->clear_args_offset();
  ctx->set_args_offset("args111111", 10);
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), PARAM_INVALID);

  ctx->clear_args_offset();
  ctx->set_op_index(0);

  const char task[] = "opattr";
  AttrUtils::SetBytes(model.op_list_[0], ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), PARAM_INVALID);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_args_offset("\0\0");
  kernel_task_info.davinci_model_ = &model;

  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), PARAM_INVALID);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  context->clear_args_offset();
  context->set_args_offset("args111111", 10);
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), FAILED);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed2) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;

  context->set_args_offset("\0\0");
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  // AttrUtils::GetBytes  -> true
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), PARAM_INVALID);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed3) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;

  context->set_args_offset("\0\0");
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), PARAM_INVALID);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed4) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  const char task[] = "opattr";
  AttrUtils::SetBytes(model.op_list_[0], ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;

  context->set_args_offset("args111111", 10);
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  // rtMalloc RT_ERROR_NONE
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), FAILED);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed5) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  const char task[] = "opattr";
  AttrUtils::SetBytes(model.op_list_[0], ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;

  context->set_args_offset("args111111", 10);
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  // rtMalloc RT_ERROR_NONE
  // rtMemcpy RT_ERROR_INVALID_VALIUE
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), FAILED);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test InitAICPUCustomTask failed
TEST_F(UtestKernelTaskInfo, init_aicpu_custom_task_failed6) {
  DavinciModel model(0, nullptr);

  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  const char task[] = "opattr";
  AttrUtils::SetBytes(model.op_list_[0], ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  kernel_task_info.davinci_model_ = &model;

  context->set_args_offset("args111111", 10);
  // KernelTaskInfo::StoreInputOutputTensor   -> SUCCESS
  // rtMalloc RT_ERROR_NONE
  // rtMemcpy RT_ERROR_NONE
  EXPECT_EQ(kernel_task_info.InitAICPUCustomTask(0, *kernel_def), FAILED);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  task_def.set_type(RT_MODEL_TASK_KERNEL);
  string args;
  args.append(100, '1');
  kernel_def->set_so_name("libDvpp.so");
  kernel_def->set_kernel_name("DvppResize");
  kernel_def->set_args(args.data(), 100);
  kernel_def->set_args_size(100);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(6);
  ctx->set_op_index(0);

  // ModelUtils::GetInputDataAddrs  -> ok
  // ModelUtils::GetOutputDataAddrs -> ok
  // rtMalloc -> RT_ERROR_NONE
  // rtMemcpy -> RT_ERROR_NONE
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type_fail) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");
  AttrUtils::SetBool(model.op_list_[0], "_AllShape", true);

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  task_def.set_type(RT_MODEL_TASK_KERNEL);
  string args;
  args.append(100, '1');
  kernel_def->set_so_name("libDvpp.so");
  kernel_def->set_kernel_name("DvppResize");
  kernel_def->set_args(args.data(), 100);
  kernel_def->set_args_size(100);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(6);
  ctx->set_op_index(0);

  // ModelUtils::GetInputDataAddrs  -> ok
  // ModelUtils::GetOutputDataAddrs -> ok
  // rtMalloc -> RT_ERROR_NONE
  // rtMemcpy -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), SUCCESS);

  const string ext_info = {1, 1, 1, 1, 0, 0, 0, 0};
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(ext_info), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, init_kernel_taskInfo_with_aicpu_kernel_type_fail2) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  task_def.set_type(RT_MODEL_TASK_KERNEL);
  string args;
  args.append(100, '1');
  kernel_def->set_so_name("libDvpp.so");
  kernel_def->set_kernel_name("DvppResize");
  kernel_def->set_args(args.data(), 100);
  kernel_def->set_args_size(100);

  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(6);
  ctx->set_op_index(0);

  // ModelUtils::GetInputDataAddrs  -> ok
  // ModelUtils::GetOutputDataAddrs -> ok
  // rtMalloc -> RT_ERROR_INVALID_VALUE
  // rtMemcpy -> RT_ERROR_NONE
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), SUCCESS);

  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test StoreInputOutputTensor failed
TEST_F(UtestKernelTaskInfo, store_input_output_tensor_fail) {
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<::tagCcAICPUTensor> input_descs;
  std::vector<::tagCcAICPUTensor> output_descs;

  KernelTaskInfo kernel_task_info;
  // rtMalloc -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);
}


TEST_F(UtestKernelTaskInfo, store_input_output_tensor_fail2) {
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<::tagCcAICPUTensor> input_descs;
  std::vector<::tagCcAICPUTensor> output_descs;

  KernelTaskInfo kernel_task_info;
  // rtMalloc -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);
}

// test InitCceTask success
TEST_F(UtestKernelTaskInfo, kernel_task_info_init_cce_task) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));

  model.runtime_param_.logic_mem_base = 0;
  model.runtime_param_.mem_size = 0;
  model.runtime_param_.logic_weight_base = 0;
  model.runtime_param_.weight_size = 0;
  model.runtime_param_.logic_var_base = 0;
  model.runtime_param_.var_size = 0;

  // KernelTaskInfo::UpdateCceArgs -> SUCCESS
  // KernelTaskInfo::UpdateCceArgs -> SUCCESS
  // rtMalloc -> RT_ERROR_NONE
  // rtMemcpy -> RT_ERROR_NONE
  // rtMemAllocManaged  -> RT_ERROR_NONE
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed1) {
  DavinciModel model(0, nullptr);

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed2) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed3) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  // KernelTaskInfo::UpdateCceArgs  -> CCE_FAILED
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed4) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  // KernelTaskInfo::UpdateCceArgs  -> SUCCESS
  // KernelTaskInfo::SetFlowtable  -> RT_FAILED
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed5) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  // KernelTaskInfo::UpdateCceArgs  -> SUCCESS
  // KernelTaskInfo::SetFlowtable  -> SUCCESS
  // rtMalloc  -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed6) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  // KernelTaskInfo::UpdateCceArgs  -> SUCCESS
  // KernelTaskInfo::SetFlowtable  -> SUCCESS
  // rtMalloc  -> RT_ERROR_NONE
  // rtMemcpy  -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_cce_task_failed7) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("", "");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  // KernelTaskInfo::SetContext  -> SUCCESS

  kernel_def->set_flowtable("InitCceTask");
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_is_flowtable(true);

  // KernelTaskInfo::UpdateCceArgs  -> SUCCESS
  // KernelTaskInfo::SetFlowtable  -> SUCCESS
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernel_def->set_sm_desc(&l2CtrlInfo, sizeof(rtSmDesc_t));

  // rtMalloc  -> RT_ERROR_NONE
  // rtMemcpy  -> RT_ERROR_NONE
  // rtMemAllocManaged -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.InitCceTask(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test SetContext success
TEST_F(UtestKernelTaskInfo, success_kernel_taskInfo_init_set_context) {
  DavinciModel model(0, nullptr);

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(1);
  context->set_args_offset("args111111", 10);

  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_task_info.SetContext(*kernel_def), SUCCESS);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test SetContext failed
TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_set_context_failed1) {
  DavinciModel model(0, nullptr);

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(0);

  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_task_info.SetContext(*kernel_def), INTERNAL_ERROR);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_taskInfo_init_set_context_failed2) {
  DavinciModel model(0, nullptr);

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_id(1);
  context->set_kernel_func_id(1);
  context->set_is_flowtable(true);
  context->set_args_count(5);
  context->set_args_offset("\0\0");  // args_offset = 0

  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  EXPECT_EQ(kernel_task_info.SetContext(*kernel_def), PARAM_INVALID);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

// test UpdateCceArgs success
TEST_F(UtestKernelTaskInfo, kernel_task_info_update_cce_args) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("InitCceTask");
  string sm_desc("args");

  uint8_t test = 2;
  model.mem_base_ = &test;
  model.runtime_param_.logic_mem_base = 0;

  model.weights_mem_base_ = &test;
  model.runtime_param_.logic_weight_base = 0;

  uint8_t test1 = 16;
  model.var_mem_base_ = &test1;
  model.runtime_param_.logic_var_base = 0;

  context->set_is_flowtable(true);
  // KernelTaskInfo::CceUpdateKernelArgs ->SUCCESS
  EXPECT_EQ(kernel_task_info.UpdateCceArgs(sm_desc, flowtable, *kernel_def), FAILED);


  context->clear_is_flowtable();
  context->set_is_flowtable(false);
  // KernelTaskInfo::CceUpdateKernelArgs ->SUCCESS
  EXPECT_EQ(kernel_task_info.UpdateCceArgs(sm_desc, flowtable, *kernel_def), FAILED);

  kernel_def->clear_context();
  task_def.clear_kernel();

  model.mem_base_ = nullptr;
  model.weights_mem_base_ = nullptr;
  model.var_mem_base_ = nullptr;
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_update_cce_args_failed1) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("InitCceTask");
  string sm_desc("args");

  uint8_t test = 2;
  model.mem_base_ = &test;
  model.runtime_param_.logic_mem_base = 0;

  uint8_t test1 = 10;
  model.weights_mem_base_ = &test1;
  model.runtime_param_.logic_weight_base = 0;

  model.var_mem_base_ = &test1;
  model.runtime_param_.logic_var_base = 0;

  context->set_is_flowtable(true);
  // KernelTaskInfo::CceUpdateKernelArgs -> FAILED
  EXPECT_EQ(kernel_task_info.UpdateCceArgs(sm_desc, flowtable, *kernel_def), FAILED);

  kernel_def->clear_context();
  task_def.clear_kernel();

  model.mem_base_ = nullptr;
  model.weights_mem_base_ = nullptr;
  model.var_mem_base_ = nullptr;
}

// test SetFlowtable
TEST_F(UtestKernelTaskInfo, kernel_task_info_set_flowtable) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("InitCceTask");
  context->set_is_flowtable(false);
  EXPECT_EQ(kernel_task_info.SetFlowtable(flowtable, *kernel_def), SUCCESS);


  context->clear_is_flowtable();
  context->set_is_flowtable(true);
  // rtMalloc ->RT_ERROR_NONE
  // rtMemcpy ->RT_ERROR_NONE
  kernel_def->set_args("args111111", 10);
  context->set_args_offset("\0\0");
  EXPECT_EQ(kernel_task_info.SetFlowtable(flowtable, *kernel_def), SUCCESS);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_set_flowtable_failed1) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("SetFlowtable");
  context->set_is_flowtable(true);

  // rtMalloc -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.SetFlowtable(flowtable, *kernel_def), FAILED);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_set_flowtable_failed2) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("SetFlowtable");
  context->set_is_flowtable(true);
  // rtMalloc ->RT_ERROR_NONE
  // rtMemcpy ->RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.SetFlowtable(flowtable, *kernel_def), FAILED);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_set_flowtable_failed3) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  model.stream_list_ = { stream };
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = model.op_list_[0];

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();

  string flowtable("SetFlowtable");
  context->set_is_flowtable(true);
  // rtMalloc ->RT_ERROR_NONE
  // rtMemcpy ->RT_ERROR_NONE
  kernel_def->set_args("args", 4);
  context->set_args_offset("args111111", 10);
  EXPECT_EQ(kernel_task_info.SetFlowtable(flowtable, *kernel_def), FAILED);

  kernel_def->clear_context();
  task_def.clear_kernel();
}

TEST_F(UtestKernelTaskInfo, distribute_failed) {
  KernelTaskInfo kernel_task_info;
  DavinciModel model(0, nullptr);

  domi::TaskDef task_def;

  // Failed for SetStream
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);

  // rtKernelLaunchWithFlag -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, distribute_success) {
  KernelTaskInfo kernel_task_info;
  DavinciModel model(0, nullptr);
  model.op_list_[0] = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::TaskDef task_def;
  // rtModelGetTaskId -> RT_ERROR_INVALID_VALUE
  rtModel_t rt_model_handle = (rtModel_t *)0x12345678;
  model.rt_model_handle_ = rt_model_handle;

  // Failed for SetStream
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);

  // rtKernelLaunchWithFlag -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  model.rt_model_handle_ = nullptr;
}

// test success DistributeDumpTask
TEST_F(UtestKernelTaskInfo, success_distribute_dump_task) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  domi::KernelDef *kernel_def = task_def.mutable_kernel();

  kernel_def->set_stub_func("kerneltaskinfo");
  kernel_def->set_block_dim(10);
  kernel_def->set_args("args111111", 10);
  kernel_def->set_args_size(10);
  rtSmDesc_t l2CtrlInfo;
  l2CtrlInfo.data[0].L2_mirror_addr = 1024;
  kernel_def->set_sm_desc((void *)&l2CtrlInfo, sizeof(rtSmDesc_t));

  // for SetStream
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  std::vector<rtStream_t> stream_list = { stream };
  EXPECT_EQ(kernel_task_info.SetStream(0, stream_list), SUCCESS);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);

  rtStreamDestroy(stream);
  task_def.clear_kernel();
}

// test success GetTaskID
TEST_F(UtestKernelTaskInfo, success_get_task_id) {
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(RT_MODEL_TASK_KERNEL);
  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task->type()));

  EXPECT_EQ(task_info->GetTaskID(), 0);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.GetTaskID(), 0);

  HcclTaskInfo hccl_task_info;
  EXPECT_EQ(hccl_task_info.GetTaskID(), 0);
}

// test StoreInputOutputTensor success
TEST_F(UtestKernelTaskInfo, success_store_input_output_tensor) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<::tagCcAICPUTensor> input_descs;
  std::vector<::tagCcAICPUTensor> output_descs;

  int test = 1;
  int *addr = &test;
  void *input = addr;
  void *output = addr;
  input_data_addrs.push_back(input);
  output_data_addrs.push_back(output);

  tagCcAICPUTensor input_desc;
  tagCcAICPUTensor output_desc;
  input_descs.push_back(input_desc);
  output_descs.push_back(output_desc);

  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);

  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
}

// test KernelTaskInfo release fail
TEST_F(UtestKernelTaskInfo, fail_release) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<::tagCcAICPUTensor> input_descs;
  std::vector<::tagCcAICPUTensor> output_descs;

  int test = 1;
  int *addr = &test;
  void *input = addr;
  void *output = addr;
  input_data_addrs.push_back(input);
  output_data_addrs.push_back(output);

  tagCcAICPUTensor input_desc;
  tagCcAICPUTensor output_desc;
  input_descs.push_back(input_desc);
  output_descs.push_back(output_desc);

  EXPECT_EQ(kernel_task_info.StoreInputOutputTensor(input_data_addrs, output_data_addrs, input_descs, output_descs), SUCCESS);

  // rtMemFreeManaged -> RT_ERROR_INVALID_VALUE
  EXPECT_EQ(kernel_task_info.Release(), SUCCESS);
}

// test KernelTaskInfo release fail
TEST_F(UtestKernelTaskInfo, update_l2data_success) {
  DavinciModel model(0, nullptr);
  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  domi::KernelDef kernel_def;

  EXPECT_EQ(kernel_task_info.UpdateL2Data(kernel_def), SUCCESS);
}

// test fusion_end_task Init
TEST_F(UtestKernelTaskInfo, kernel_task_info_init_success) {
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);

  DavinciModel model(0, nullptr);
  auto model_def = MakeShared<domi::ModelTaskDef>();

  model.model_id_ = 1;
  model.name_ = "test";
  model.version_ = 0x01;

  model.stream_list_ = { stream };
  model.ge_model_ = MakeShared<GeModel>();
  model.ge_model_->SetModelTaskDef(model_def);

  auto op_desc = CreateOpDesc("data", DATA);
  op_desc->SetInputOffset({1});
  op_desc->SetOutputOffset({100});

  GeTensorDesc descin(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(descin, 4);
  op_desc->AddInputDesc(descin);
  GeTensorDesc descout(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  TensorUtils::SetSize(descout, 32);
  op_desc->AddOutputDesc(descout);
  op_desc->SetId(0);

  model.op_list_[0] = op_desc;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_op_index(0);
  vector<string> original_op_names = { "conv", "add" };
  AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.Init(task_def, &model), FAILED);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_te) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(2);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.CalculateArgs(task_def, &model), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_aicpu) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(6);

  KernelTaskInfo kernel_task_info;
  EXPECT_EQ(kernel_task_info.CalculateArgs(task_def, &model), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_calculate_args_custom_aicpu) {
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;

  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  domi::KernelContext *ctx = kernel_def->mutable_context();
  ctx->set_kernel_type(7);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.kernel_type_ = ccKernelType::CUST_AI_CPU;
  kernel_task_info.op_desc_ = std::make_shared<OpDesc>("concat", "TensorArrayWrite");
  kernel_task_info.InitDumpArgs(0);
  EXPECT_EQ(kernel_task_info.CalculateArgs(task_def, &model), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_update_args_te) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  EXPECT_EQ(kernel_task_info.UpdateArgs(), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, kernel_task_info_update_args_aicpu) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.kernel_type_ = ccKernelType::TE;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");
  kernel_task_info.args_size_ = 120;
  kernel_task_info.args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[kernel_task_info.args_size_]);
  kernel_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  rtMalloc(&kernel_task_info.args_, kernel_task_info.args_size_, RT_MEMORY_HBM);

  EXPECT_EQ(kernel_task_info.UpdateArgs(), SUCCESS);
}


TEST_F(UtestKernelTaskInfo, kernel_task_info_super_kernel_info) {
  DavinciModel model(0, nullptr);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = CreateOpDesc("FrameworkOp", "FrameworkOp");

  EXPECT_EQ(kernel_task_info.SaveSuperKernelInfo(), SUCCESS);

  kernel_task_info.UpdateSKTTaskId();

  EXPECT_EQ(kernel_task_info.SKTFinalize(), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op) {
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
  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  op_desc->SetId(0);
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
  kernel_task_info.op_desc_ = op_desc;
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op_fail_01) {
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

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  op_desc->SetId(0);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &davinci_model;
  kernel_task_info.op_desc_ = op_desc;

  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);

  kernel_task_info.is_blocking_aicpu_op_ = true;
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);
}

TEST_F(UtestKernelTaskInfo, blocking_aicpu_op_fail_02) {
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

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);

  const OpDescPtr op_desc = CreateOpDesc("deque", "Deque");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  op_desc->SetId(0);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_.emplace(0, op_desc);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &davinci_model;
  kernel_task_info.op_desc_ = op_desc;

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);

  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtEventReset, rtError_t, 0x78000001);
  EXPECT_EQ(kernel_task_info.Distribute(), FAILED);

  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_task_info.InitAicpuTaskExtInfo(kernel_def.kernel_ext_info()), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  EXPECT_EQ(kernel_task_info.Distribute(), SUCCESS);
}

}  // namespace ge
