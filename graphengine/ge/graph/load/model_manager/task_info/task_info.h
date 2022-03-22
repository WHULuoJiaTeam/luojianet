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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_

#include <vector>
#include <sstream>

#include "cce/customize.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/ts_mem_mall.h"
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "proto/task.pb.h"

namespace ge {
struct MemInfo {
  size_t memory_size = 0;
  uint64_t logic_memory_base = 0;
  uint8_t *memory_base = nullptr;
  uint32_t memory_type = RT_MEMORY_HBM;
  std::string memory_key = "";
};

struct RuntimeParam {
  RuntimeParam() {
    ts_mem_mall = std::unique_ptr<TsMemMall>(new (std::nothrow) TsMemMall());
    aicpu_mem_mall = std::unique_ptr<TsMemMall>(new (std::nothrow) TsMemMall(RT_MEMORY_HBM));
  }
  ~RuntimeParam() = default;

  std::string ToString() {
    std::stringstream ss;
    ss << "session_id:" << session_id << ", stream_num:" << stream_num << ", event_num:" << event_num
       << ", label_num:" << label_num << ", logic_mem_base:" << logic_mem_base
       << ", logic_weight_base:" << logic_weight_base << ", logic_var_base:" << logic_var_base
       << ", memory_size:" << mem_size << ", weight_size:" << weight_size << ", var_size:" << var_size
       << ", zero_copy_size:" << zero_copy_size << ", ex_memory_info:";
    for (auto it : memory_infos) {
      ss << "[memory_type:" << it.first << ", memory_size:" << it.second.memory_size << "]";
    }
    return ss.str();
  }

  uint64_t mem_size = 0;
  uint64_t logic_mem_base = 0;
  uint8_t *mem_base = nullptr;
  uint64_t weight_size = 0;
  uint64_t logic_weight_base = 0;
  uint8_t *weight_base = nullptr;
  uint64_t var_size = 0;
  uint64_t logic_var_base = 0;
  uint8_t *var_base = nullptr;
  int64_t zero_copy_size = 0;
  std::map<uint64_t, MemInfo> memory_infos;
  uint32_t batch_num = 0;
  uint32_t stream_num = 0;
  uint32_t event_num = 0;
  uint32_t label_num = 0;
  uint64_t session_id = 0;
  uint32_t graph_id = 0;
  bool is_single_op = false;

  std::unique_ptr<TsMemMall> ts_mem_mall;
  std::unique_ptr<TsMemMall> aicpu_mem_mall;
};

typedef struct FusionOpInfo {
  std::vector<std::string> original_op_names;
  std::string op_name;
  uint32_t op_index;
  uint32_t stream_id;
} FusionOpInfo;

class DavinciModel;

class TaskInfo {
 public:
  TaskInfo() : stream_(nullptr) {}

  virtual ~TaskInfo() { stream_ = nullptr; }

  virtual Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) = 0;

  virtual Status Distribute() = 0;

  virtual Status UpdateArgs() { return SUCCESS; }

  virtual Status CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) { return SUCCESS; }

  virtual Status Release() { return SUCCESS; }

  virtual ccOpContext *GetCtx() { return nullptr; }

  virtual uint32_t GetTaskID() { return 0xFFFFFFFF; }

  virtual bool CallSaveDumpInfo() { return false; }

  virtual uint32_t GetStreamId() { return 0xFFFFFFFF; }

  virtual uintptr_t GetDumpArgs() { return 0; }

  virtual uint32_t GetSktTaskID() { return 0xFFFFFFFF; }

  virtual FusionOpInfo *GetFusionOpInfo() { return nullptr; }

 protected:
  Status SetStream(uint32_t stream_id, const std::vector<rtStream_t> &stream_list);

  void *stream_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_H_
