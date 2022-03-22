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

#include "graph/load/model_manager/task_info/ffts_task_info.h"
#include "cce/aicpu_engine_struct.h"
#include "common/ge/ge_util.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/attr_value.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "runtime/rt_ffts.h"

namespace ge {
extern OpDescPtr CreateOpDesc(string name, string type);

class UtestFftsTaskInfo : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}

public:
  void CreateFftsTaskInfo(DavinciModel &davinci_model, domi::TaskDef &task_def, FftsTaskInfo &ffts_task_info) {
    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);
    davinci_model.stream_list_ = { stream };
    task_def.set_stream_id(0);

    domi::FftsTaskDef *ffts_task_def = task_def.mutable_ffts_task();
    davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
    ffts_task_def->set_op_index(0);
    ffts_task_def->set_addr_size(2);
    domi::FftsDescInfoDef *ffts_desc = ffts_task_def->mutable_ffts_desc();
    ffts_desc->set_tm(0);
    rtFftsTaskInfo_t sub_task_info;
    ffts_task_info.sub_task_info_ = sub_task_info;
    ffts_task_def->set_ffts_type(RT_FFTS_TYPE_AUTO_THREAD);
  }
};

// test FftsTaskInfo Init with no subtask and no ticket cache
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_without_subtask) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  davinci_model.stream_list_ = { stream };
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::FftsTaskDef *ffts_task_def = task_def.mutable_ffts_task();
  FftsTaskInfo ffts_task_info;
  // init failed when model without op_desc
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), PARAM_INVALID);

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_task_def->set_op_index(0);
  ffts_task_def->set_addr_size(2);
  domi::FftsDescInfoDef *ffts_desc = ffts_task_def->mutable_ffts_desc();
  ffts_desc->set_tm(0);
  rtFftsTaskInfo_t sub_task_info;
  ffts_task_info.sub_task_info_ = sub_task_info;
  ffts_task_def->set_ffts_type(RT_FFTS_TYPE_AUTO_THREAD);
  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Init with subtask and no ticket cache: AutoThreadAicAivDef
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_with_auto_thread_subgraph) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  CreateFftsTaskInfo(davinci_model, task_def, ffts_task_info);
  domi::FftsSubTaskDef *ffts_sub_task_def = task_def.mutable_ffts_task()->add_sub_task();
  ffts_sub_task_def->set_thread_dim(static_cast<uint32_t>(1));
  //sub_task_def.has_auto_thread_aic_aiv() == sub_task_def.has_manual_thread_aic_aiv()
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), FAILED);

  domi::AutoThreadAicAivDef *auto_thread_aic_aiv_def = ffts_sub_task_def->mutable_auto_thread_aic_aiv();
  domi::AutoThreadPrefetchDef *src_prefetch = auto_thread_aic_aiv_def->add_src_prefetch();
  // without InitIoAddrs
  ffts_task_info.thread_dim_ = 0;
  RuntimeParam runtime_param;
  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Init with subtask and no ticket cache: ManualThreadAicAivDef
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_with_manual_thread_subgraph) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  CreateFftsTaskInfo(davinci_model, task_def, ffts_task_info);
  domi::FftsSubTaskDef *ffts_sub_task_def = task_def.mutable_ffts_task()->add_sub_task();
  ffts_sub_task_def->set_thread_dim(static_cast<uint32_t>(1));
  //sub_task_def.has_auto_thread_aic_aiv() == sub_task_def.has_manual_thread_aic_aiv()

  domi::ManualThreadAicAivDef *manual_thread_aic_aiv_def = ffts_sub_task_def->mutable_manual_thread_aic_aiv();
  manual_thread_aic_aiv_def->add_thread_prefetch_dmu_idx(static_cast<uint32_t>(0));
  manual_thread_aic_aiv_def->add_thread_blk_dim(static_cast<uint32_t>(0));
  manual_thread_aic_aiv_def->add_thread_task_func_stub("ffts");
  domi::ManualThreadDmuDef *prefetch_list = manual_thread_aic_aiv_def->add_prefetch_list();
  prefetch_list->set_data_addr(static_cast<uint64_t>(0));
  // without InitIoAddrs
  ffts_task_info.thread_dim_ = 0;
  RuntimeParam runtime_param;
  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Init with subtask and no ticket cache: ManualThreadNopDef
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_with_manual_thread_nop_subgraph) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  CreateFftsTaskInfo(davinci_model, task_def, ffts_task_info);

  domi::FftsSubTaskDef *ffts_sub_task_def = task_def.mutable_ffts_task()->add_sub_task();
  ffts_sub_task_def->set_thread_dim(static_cast<uint32_t>(1));
  domi::AutoThreadAicAivDef *auto_thread_aic_aiv_def = ffts_sub_task_def->mutable_auto_thread_aic_aiv();
  domi::ManualThreadNopDef *manual_thread_nop = ffts_sub_task_def->mutable_manual_thread_nop();
  domi::ManualThreadDependencyDef *src_dep_tbl = manual_thread_nop->add_src_dep_tbl();
  src_dep_tbl->add_dependency(static_cast<uint32_t>(0));

  // without InitIoAddrs
  ffts_task_info.thread_dim_ = 0;
  RuntimeParam runtime_param;
  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Init with no subtask and ticket cache:AutoThreadCacheDef
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_with_auto_thread_ticket_cache) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  CreateFftsTaskInfo(davinci_model, task_def, ffts_task_info);

  domi::TicketCacheDef *ticket_cache_def = task_def.mutable_ffts_task()->add_ticket_cache();
  //ticket_cache_def.has_auto_thread_cache() == ticket_cache_def.has_manual_thread_cache()
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), FAILED);
  domi::AutoThreadCacheDef *auto_thread_cache = ticket_cache_def->mutable_auto_thread_cache();

  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Init with no subtask and ticket cache:ManualThreadCacheDef
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_with_manual_thread_ticket_cache) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  CreateFftsTaskInfo(davinci_model, task_def, ffts_task_info);

  domi::TicketCacheDef *ticket_cache_def = task_def.mutable_ffts_task()->add_ticket_cache();
  domi::ManualThreadCacheDef *manual_thread_cache = ticket_cache_def->mutable_manual_thread_cache();
  manual_thread_cache->add_slice_dmu_idx(static_cast<uint32_t>(0));
  manual_thread_cache->add_ticket_cache_ref_cnt_tbl(static_cast<uint32_t>(0));
  domi::ManualThreadDmuDef  *dmu_list = manual_thread_cache->add_dmu_list();

  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.Init(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo UpdateArgs
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_update_args) {
  DavinciModel davinci_model(0, nullptr);
  FftsTaskInfo ffts_task_info;
  ffts_task_info.davinci_model_ = &davinci_model;
  ffts_task_info.io_addrs_ = { (void*)0x12345678, (void*)0x22345678 };
  EXPECT_EQ(ffts_task_info.UpdateArgs(), SUCCESS);
}

// test FftsTaskInfo CalculateArgs
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_calculate_args) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsTaskInfo ffts_task_info;
  EXPECT_EQ(ffts_task_info.CalculateArgs(task_def, &davinci_model), SUCCESS);
}

// test FftsTaskInfo Distribute
TEST_F(UtestFftsTaskInfo, success_ffts_task_info_distribute) {
  DavinciModel davinci_model(0, nullptr);
  FftsTaskInfo ffts_task_info;
  rtFftsTaskInfo_t sub_task_info;
  ffts_task_info.sub_task_info_ = sub_task_info;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ffts_task_info.stream_ = stream;
  EXPECT_EQ(ffts_task_info.Distribute(), SUCCESS);
}
}  // namespace ge