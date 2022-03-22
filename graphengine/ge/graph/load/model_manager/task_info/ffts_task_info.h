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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FFTS_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FFTS_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class FftsTaskInfo : public TaskInfo {
 public:
  FftsTaskInfo() = default;
  ~FftsTaskInfo() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status UpdateArgs() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

 private:
  void InitFftsDescInfo(const domi::FftsDescInfoDef &ffts_desc_def, rtFftsDescInfo_t &ffts_desc);
  Status InitSubTaskInfo(const domi::FftsSubTaskDef &task_def, rtFftsSubTaskInfo_t &task);
  Status InitTicketCache(const domi::TicketCacheDef &cache_def, rtTicketCache_t &cache);

  Status InitAutoAicAiv(const domi::AutoThreadAicAivDef &aic_aiv_def, rtAutoThreadAicAivInfo_t &aic_aiv);
  void InitAutoCacheInfo(const domi::AutoThreadCacheDef &cache_def, rtAutoThreadCacheInfo_t &cache);
  void InitAutoPrefetch(const domi::AutoThreadPrefetchDef &prefetch_def, rtAutoThreadPrefetch_t &prefetch);

  Status InitManualAicAiv(const domi::ManualThreadAicAivDef &aic_aiv_def, rtManualThreadAicAivInfo_t &aic_aiv);
  Status InitManualCacheInfo(const domi::ManualThreadCacheDef &cache_def, rtManualThreadCacheInfo_t &cache);
  Status InitManualDependency(const domi::ManualThreadDependencyDef &depend_def, rtManualThreadDependency_t &depend);
  Status InitManualNop(const domi::ManualThreadNopDef &nop_def, rtManualThreadNopInfo_t &nop);

  void InitManualDmuInfo(const domi::ManualThreadDmuDef &dmu_def, rtManualThreadDmuInfo_t &dmu);
  void InitManualDmuInfo(const domi::ManualThreadCacheDef &cache_def, rtManualThreadDmuInfo_t *&dmu);
  void InitManualDmuInfo(const domi::ManualThreadAicAivDef &aic_aiv_def, rtManualThreadDmuInfo_t *&dmu);

  template<typename T>
  Status InitIoAddrs(const RuntimeParam &rts_param, const T &aic_aiv_def, uint32_t thread_dim, uint32_t addr_count);

  DavinciModel *davinci_model_{nullptr};
  rtFftsTaskInfo_t sub_task_info_;
  std::vector<void *> io_addrs_;
  uint32_t thread_dim_{0};
  void *args_{nullptr};    // runtime args memory
  uint32_t args_size_{0};    // runtime args memory length
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FFTS_TASK_INFO_H_
