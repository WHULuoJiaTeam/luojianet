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

#ifndef GE_COMMON_PROFILING_PROFILING_MANAGER_H_
#define GE_COMMON_PROFILING_PROFILING_MANAGER_H_

#include <nlohmann/json.hpp>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "external/register/register_types.h"
#include "toolchain/prof_callback.h"
#include "runtime/stream.h"

using std::map;
using std::string;
using std::vector;
using Json = nlohmann::json;

namespace {
  const std::string GE_PROFILING_MODULE = "Framework";
  // DataTypeConfig MASK
  const uint64_t PROF_ACL_API_MASK = 0x0001;
  const uint64_t PROF_TASK_TIME_MASK = 0x0002;
  const uint64_t PROF_AICORE_METRICS_MASK = 0x0004;
  const uint64_t PROF_AICPU_TRACE_MASK = 0x0008;
  const uint64_t PROF_MODEL_EXECUTE_MASK = 0x0010;
  const uint64_t PROF_RUNTIME_API_MASK = 0x0020;
  const uint64_t PROF_RUNTIME_TRACE_MASK = 0x0040;
  const uint64_t PROF_SCHEDULE_TIMELINE_MASK = 0x0080;
  const uint64_t PROF_SCHEDULE_TRACE_MASK = 0x0100;
  const uint64_t PROF_AIVECTORCORE_METRICS_MASK = 0x0200;
  const uint64_t PROF_SUBTASK_TIME_MASK = 0x0400;
  const uint64_t PROF_TRAINING_TRACE_MASK = 0x0800;
  const uint64_t PROF_HCCL_TRACE_MASK = 0x1000;
  const uint64_t PROF_DATA_PROCESS_MASK = 0x2000;
  const uint64_t PROF_MODEL_LOAD_MASK = 0x8000000000000000;

}  // namespace
namespace ge {
class OpDesc;
using OpDescPtr = std::shared_ptr<OpDesc>;
struct DeviceSubsInfo {
  uint64_t module;
  uint32_t subscribe_count;
};

struct ProfSubscribeInfo {
  bool is_subscribe;
  uint64_t prof_switch;
  uint32_t graph_id;
};

struct MsprofCallback {
  MsprofCtrlCallback msprofCtrlCallback;
  MsprofReporterCallback msprofReporterCallback;
};

class ProfilingManager {
 public:
  ProfilingManager();
  virtual ~ProfilingManager();
  static ProfilingManager &Instance();
  Status Init(const Options &options);
  Status ProfInit(uint64_t module);
  Status ProfFinalize();
  Status ProfStartProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  Status ProfStopProfiling(uint64_t module, const std::map<std::string, std::string> &config_para);
  Status ProfModelSubscribe(uint64_t module, void *model);
  Status ProfModelUnsubscribe(void *model);
  void StopProfiling();
  bool ProfilingTrainingTraceOn() const { return is_training_trace_; }
  // report model load profiling data flag, data contain task desc info, step info, model load fusion op info
  bool ProfilingModelLoadOn() const { return is_load_profiling_; }
  // report model execute profiling data flag, data contain model execute time info
  bool ProfilingModelExecuteOn() const;
  // is_execute_profiling_ only used by ge option and env
  bool ProfilingOn() const { return is_load_profiling_ && is_execute_profiling_; }
  void ReportProfilingData(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info);
  void ProfilingTaskDescInfo(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info,
                             const int32_t &device_id);
  void ProfilingOpInputOutInfo(const TaskDescInfo &task, Json &task_json);
  Status PluginInit();
  void PluginUnInit() const;
  Status CallMsprofReport(ReporterData &reporter_data) const;
  struct MsprofCallback &GetMsprofCallback() { return prof_cb_; }
  void SetMsprofCtrlCallback(MsprofCtrlCallback func) { prof_cb_.msprofCtrlCallback = func; }
  void SetMsprofReporterCallback(MsprofReporterCallback func) { prof_cb_.msprofReporterCallback = func; }
  void GetFpBpPoint(std::string &fp_point, std::string &bp_point);
  void GetOpInputOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;
  void ReportData(const int32_t &device_id, const std::string &data, const std::string &tag_name);
  Status ProfileStepInfo(uint64_t index_id, uint64_t model_id, uint16_t tag_id, rtStream_t stream, int32_t device_id);
  void SetStepInfoIndex(uint64_t index_id) { index_id_ = index_id; }
  uint64_t GetStepInfoIndex() const { return index_id_; }
  void SetGraphIdToDeviceMap(uint32_t graph_id, uint32_t device_id) { device_id_map_[graph_id] = device_id; }
  Status GetDeviceIdFromGraph(uint32_t graph_id, uint32_t &device_id);
  void SetSubscribeInfo(uint64_t prof_switch, uint32_t model_id, bool is_subscribe);
  const ProfSubscribeInfo &GetSubscribeInfo() const { return subscribe_info_; }
  void CleanSubscribeInfo();
  void SetGraphIdToModelMap(uint32_t graph_id, uint32_t model_id) { model_id_map_[graph_id] = model_id; }
  Status GetModelIdFromGraph(uint32_t graph_id, uint32_t &model_id);

 private:
  Status InitFromOptions(const Options &options, MsprofGeOptions &prof_conf);
  Status ParseOptions(const std::string &options);
  Status ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                        vector<int32_t> &device_list);
  Status ProfParseDeviceId(const std::map<std::string, std::string> &config_para,
                               vector<int32_t> &device_list);
  uint64_t GetProfilingModule();
  void UpdateDeviceIdModuleMap(string prof_type, uint64_t module, const vector<int32_t> &device_list);
  void UpdateSubscribeDeviceModuleMap(std::string prof_type, uint32_t device_id, uint64_t module);
  void GetOpInputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;
  void GetOpOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;

  bool is_load_profiling_;
  bool is_execute_profiling_;
  bool is_training_trace_;
  vector<int32_t> device_id_;
  map<int32_t, uint64_t> device_id_module_map_; // key: device_id, value: profiling on module
  map<uint32_t, DeviceSubsInfo> subs_dev_module_; // key: device_id, value: profiling on module
  uint32_t subscribe_count_;
  std::mutex mutex_;
  std::mutex mutex_report_;
  MsprofCallback prof_cb_;
  std::string fp_point_;
  std::string bp_point_;
  uint32_t reporter_max_len_ = 0;
  uint64_t index_id_;
  std::map<uint32_t, uint32_t> device_id_map_; // key: graph_id, value: device_id
  std::map<uint32_t, uint32_t> model_id_map_; // key: graph_id, value: model_id
  ProfSubscribeInfo subscribe_info_;
};
}  // namespace ge
#endif  // GE_COMMON_PROFILING_PROFILING_MANAGER_H_
