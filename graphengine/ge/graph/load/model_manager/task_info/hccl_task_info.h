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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/opskernel/ge_task_info.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/manager/util/hcom_util.h"
namespace ge {
class HcclTaskInfo : public TaskInfo {
 public:
  HcclTaskInfo()
      : davinci_model_(nullptr),
        id_(0),
        hccl_stream_list_(),
        ops_kernel_store_(nullptr),
        private_def_(nullptr),
        private_def_len_(0),
        args_(nullptr),
        args_offset_(0) {}

  ~HcclTaskInfo() override;

  ge::Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  ge::Status Distribute() override;

  uint32_t GetTaskID() override { return id_; }

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status UpdateArgs() override;

 private:
  void SetIoAddrs(const OpDescPtr &op_desc);

  Status SetAddrs(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  void TransToGETaskInfo(GETaskInfo &ge_task);

  void GetPrivateDefByTaskDef(const domi::TaskDef &task);

  ge::Status CreateStream(int64_t stream_num, DavinciModel *davinci_model, int64_t main_stream_id);

  Status SetFollowStream(const ge::ConstOpDescPtr &op_desc, DavinciModel *davinci_model);

  void CreateKernelHcclInfo(const ge::ConstOpDescPtr &op_desc);

  Status SetWorkspace(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  DavinciModel *davinci_model_;
  uint32_t id_;
  vector<rtStream_t> hccl_stream_list_;
  void *ops_kernel_store_;
  void *private_def_;
  uint32_t private_def_len_;
  static std::mutex hccl_follow_stream_mutex_;
  vector<GETaskKernelHcclInfo> kernel_hccl_infos_;
  vector<void *> io_addrs_;
  void *args_;
  uint32_t args_offset_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
