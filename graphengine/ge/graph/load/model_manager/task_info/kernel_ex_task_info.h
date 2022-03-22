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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"

namespace ge {
class KernelExTaskInfo : public TaskInfo {
 public:
  KernelExTaskInfo()
      : task_id_(0),
        stream_id_(0),
        dump_flag_(RT_KERNEL_DEFAULT),
        kernel_buf_size_(0),
        davinci_model_(nullptr),
        kernel_buf_(nullptr),
        input_output_addr_(nullptr),
        ext_info_addr_(nullptr),
        dump_args_(nullptr) {}

  ~KernelExTaskInfo() override {}

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status Release() override;

  Status UpdateArgs() override;

  Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  uint32_t GetTaskID() override { return task_id_; }

  uint32_t GetStreamId() override { return stream_id_; }

  uintptr_t GetDumpArgs() override {
    auto ret = reinterpret_cast<uintptr_t>(dump_args_);
    return ret;
  }
  bool CallSaveDumpInfo() override {
    return true;
  };
 private:
  Status CopyTaskInfo(const domi::KernelExDef &kernel_def, const RuntimeParam &rts_param, const OpDescPtr &op_desc);
  void SetIoAddrs(const OpDescPtr &op_desc);

  void InitDumpFlag(const OpDescPtr &op_desc);
  void InitDumpArgs(void *addr, const OpDescPtr &op_desc);
  Status InitTaskExtInfo(const std::string &ext_info, const OpDescPtr &op_desc);

  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp();
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support);
  Status UpdateEventIdForAicpuBlockingOp(const OpDescPtr &op_desc,
                                         std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle);

  uint32_t task_id_;
  uint32_t stream_id_;
  uint32_t dump_flag_;
  uint32_t kernel_buf_size_;
  DavinciModel *davinci_model_;
  OpDescPtr op_desc_;
  void *kernel_buf_;
  void *input_output_addr_;
  void *ext_info_addr_;
  void *dump_args_;
  vector<void *> io_addrs_;
  uint32_t args_offset_ = 0;
  int64_t fixed_addr_offset_ = 0;
  int32_t topic_type_flag_ = -1;
  bool is_blocking_aicpu_op_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_
