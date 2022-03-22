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
#include "graph/load/model_manager/cpu_queue_schedule.h"
#undef private
#undef protected

using namespace std;

namespace ge {
class UtestCpuQueueSchedule : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_CpuTaskZeroCopy_succ
TEST_F(UtestCpuQueueSchedule, CpuTaskZeroCopy_Init_Success) {
  CpuTaskZeroCopy cpu_task_zero_copy(nullptr);
  std::vector<uintptr_t> mbuf_list;
  map<uint32_t, ZeroCopyOffset> outside_addrs;
  ZeroCopyOffset addr_mapping;
  addr_mapping.addr_count_ = 1;
  std::vector<void *> addr_offset;
  addr_offset.push_back((void*) 0x11110000);
  uintptr_t addr = 0x12340000;
  std::map<const void *, std::vector<void *>> outside_addr;
  outside_addr[(void*)addr] = addr_offset;
  addr_mapping.outside_addrs_.emplace_back(outside_addr);
  mbuf_list.emplace_back(addr);
  uint32_t index = 0;
  outside_addrs[index] = addr_mapping;
  EXPECT_EQ(cpu_task_zero_copy.Init(mbuf_list, outside_addrs), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskInfo_Init_args_valid) {
  CpuTaskZeroCopy cpu_task_zero_copy(nullptr);
  CpuTaskActiveEntry cpu_task_active_entry(nullptr);
  CpuTaskModelDequeue cpu_task_model_dequeue(nullptr);
  CpuTaskModelRepeat cpu_task_model_repeat(nullptr);
  CpuTaskWaitEndGraph cpu_task_wait_end_graph(nullptr);
  CpuTaskModelEnqueue cpu_task_model_enqueue(nullptr);
  CpuTaskPrepareOutput cpu_task_prepare_output(nullptr);
  EXPECT_EQ(cpu_task_zero_copy.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_active_entry.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_dequeue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_repeat.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_wait_end_graph.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_enqueue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_prepare_output.Distribute(), FAILED);
}
}  // namespace ge
