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
#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_

#include <cstdint>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/zero_copy_offset.h"
#include "runtime/kernel.h"

namespace ge {
// For AICPU task "modelDequeue" / "modelEnqueue"
struct MbufQueueInfo {
  uint32_t queue_id;        // Op queue id
  uintptr_t in_mbuf;        // addr for input mbuf
};

// For AICPU task "modelPrepareInput"
struct PrepareInputInfo {
  uintptr_t in_mbuf;        // input mbuf from dequeue
  uint32_t mbuf_offset;     // offset of mbuf(current is 0)
  uint32_t data_size;       // input Tensor size
  uintptr_t data_addr;      // input Tensor addr
};

// For AICPU task "modelPrepareOutput"
struct PrepareOutputInfo {
  uint32_t data_size;       // output Tensor size
  uintptr_t data_addr;      // output Tensor addr
  uintptr_t in_mbuf;        // input mbuf, for fill output mbuf header
  uintptr_t out_mbuf;       // output mbuf addr
};

// For AICPU task "modelZeroCopy"
struct AddrMapInfo {
  uint32_t addr_num = 0;
  uint64_t src_addr_list;
  uint64_t dst_addr_list;
};

///
/// @ingroup ge
/// @brief CpuTask base, inherit from TaskInfo used for manage.
///
class CpuTaskInfo : public TaskInfo {
 public:
  explicit CpuTaskInfo(rtStream_t stream);
  ~CpuTaskInfo() override;

 protected:
  void *args_;
  uint32_t args_size_;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
///
class CpuTaskModelDequeue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelDequeue(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelDequeue() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(uint32_t queue_id, uintptr_t &in_mbuf);

  Status Distribute() override;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, zero copy.
///
class CpuTaskZeroCopy : public CpuTaskInfo {
 public:
  explicit CpuTaskZeroCopy(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskZeroCopy() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(std::vector<uintptr_t> &mbuf_list, const map<uint32_t, ZeroCopyOffset> &outside_addrs);

  Status Distribute() override;
private:
  void *src_addr_ = nullptr;
  void *dst_addr_ = nullptr;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, active original model stream.
///
class CpuTaskPrepareOutput : public CpuTaskInfo {
 public:
  explicit CpuTaskPrepareOutput(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskPrepareOutput() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(uintptr_t addr, uint32_t size, uintptr_t in_mbuf, uintptr_t &out_mbuf);

  Status Distribute() override;
};

class CpuTaskModelEnqueue : public CpuTaskInfo {
 public:
  explicit CpuTaskModelEnqueue(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelEnqueue() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(uint32_t queue_id, uintptr_t out_mbuf);

  Status Distribute() override;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, active entry stream.
///
class CpuTaskActiveEntry : public CpuTaskInfo {
 public:
  explicit CpuTaskActiveEntry(rtStream_t stream) : CpuTaskInfo(stream), active_stream_(nullptr) {}
  ~CpuTaskActiveEntry() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(rtStream_t stream);

  Status Distribute() override;

 private:
  rtStream_t active_stream_;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
///
class CpuTaskWaitEndGraph : public CpuTaskInfo {
 public:
  explicit CpuTaskWaitEndGraph(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskWaitEndGraph() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(uint32_t model_id);

  Status Distribute() override;
};

///
/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
///
class CpuTaskModelRepeat : public CpuTaskInfo {
 public:
  explicit CpuTaskModelRepeat(rtStream_t stream) : CpuTaskInfo(stream) {}
  ~CpuTaskModelRepeat() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override { return SUCCESS; }
  Status Init(uint32_t model_id);

  Status Distribute() override;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_CPU_QUEUE_SCHEDULE_H_
