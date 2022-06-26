/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PROFILING_REPORTER_H
#define MINDSPORE_PROFILING_REPORTER_H
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "securec/include/securec.h"
#include "utils/log_adapter.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "toolchain/prof_common.h"
#include "toolchain/prof_reporter.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::pair;
using std::string;
using std::vector;

class StepPointDesc {
 public:
  StepPointDesc(string op_name, uint32_t tag) : op_name_(std::move(op_name)), tag_(tag) {}
  ~StepPointDesc() = default;

  string op_name() { return op_name_; }
  uint32_t tag() const { return tag_; }

 private:
  string op_name_;
  uint32_t tag_;
};

class ProfilingReporter {
 public:
  ProfilingReporter(int device_id, uint32_t graph_id, vector<CNodePtr> cnode_list, const vector<uint32_t> &stream_ids,
                    const vector<uint32_t> &task_ids)
      : device_id_(device_id),
        graph_id_(graph_id),
        cnode_list_(std::move(cnode_list)),
        stream_ids_(stream_ids),
        task_ids_(task_ids) {}
  ~ProfilingReporter() = default;

  void ReportTasks();
  void ReportStepPoint(const vector<std::shared_ptr<StepPointDesc>> &points);

 private:
  uint32_t device_id_;
  uint32_t graph_id_;
  vector<CNodePtr> cnode_list_;
  vector<uint32_t> stream_ids_;
  vector<uint32_t> task_ids_;
  map<string, int> node_name_index_map_;

  bool CheckStreamTaskValid() const;
  static uint32_t GetBlockDim(const CNodePtr &node);
  void ConstructNodeNameIndexMap();
  uint32_t GetStreamId(const string &node_name);
  uint32_t GetTaskId(const string &node_name);
  const CNodePtr GetCNode(const std::string &name) const;

  void ReportData(uint32_t device_id, unsigned char *data, size_t data_size, const std::string &tag_name);
  void ReportTask(const CNodePtr &node, uint32_t stream_id, uint32_t task_id, KernelType kernel_type);
  void ReportNode(const CNodePtr &node, uint32_t stream_id, uint32_t task_id, uint32_t tensor_type);
  void BuildProfTensorDataCommon(MsprofGeProfTensorData *tensor_info, uint32_t stream_id, uint32_t task_id);
  void BuildTensorData(MsprofGeTensorData *tensor_data, const CNodePtr &node, size_t index, uint32_t tensor_type);

  template <typename T>
  void SetAlternativeValue(T *property, const size_t property_size, const string &value, const int32_t &device_id) {
    MS_EXCEPTION_IF_NULL(property);
    if (value.size() < property_size) {
      property->type = static_cast<uint8_t>(MSPROF_MIX_DATA_HASH_ID);
      const auto ret = strncpy_s(property->data.dataStr, property_size, value.c_str(), value.size());
      if (ret != 0) {
        MS_LOG(ERROR) << "[Profiling] strncpy_s value " << value.c_str() << " error!";
        return;
      }
    } else {
      property->type = static_cast<uint8_t>(MSPROF_MIX_DATA_STRING);
      uint64_t hash_id;
      ProfilingManager::GetInstance().QueryHashId(device_id, value, &hash_id);
      property->data.hashId = hash_id;
    }
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_PROFILING_REPORTER_H
