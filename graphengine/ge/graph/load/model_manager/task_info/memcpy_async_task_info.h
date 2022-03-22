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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class MemcpyAsyncTaskInfo : public TaskInfo {
 public:
  MemcpyAsyncTaskInfo() : dst_(nullptr), dst_max_(0), src_(nullptr), count_(0), kind_(RT_MEMCPY_RESERVED) {}

  ~MemcpyAsyncTaskInfo() override {
    src_ = nullptr;
    dst_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status UpdateArgs() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

 private:
  Status SetIoAddrs(const OpDescPtr &op_desc, const domi::MemcpyAsyncDef &memcpy_async);

  uint8_t *dst_;
  uint64_t dst_max_;
  uint8_t *src_;
  uint64_t count_;
  uint32_t kind_;
  vector<void *> io_addrs_;
  int64_t fixed_addr_offset_ = 0;
  DavinciModel *davinci_model_ = nullptr;
  uint32_t args_offset_ = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_
