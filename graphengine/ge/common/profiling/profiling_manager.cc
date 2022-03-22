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

#include "common/profiling/profiling_manager.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/string_util.h"
#include "graph/ge_context.h"
#include "graph/utils/type_utils.h"
#include "external/graph/types.h"
#include "runtime/base.h"
#include "graph/load/model_manager/davinci_model.h"
#include "mmpa/mmpa_api.h"

namespace {
const char *const kTrainingTrace = "training_trace";
const char *const kFpPoint = "fp_point";
const char *const kBpPoint = "bp_point";

#ifdef DAVINCI_SUPPORT_PROFILING
const int32_t kMaxDeviceNum = 256;
const uint32_t kInteval = 2;
const std::string kConfigNumsdev = "devNums";
const std::string kConfigDevIdList = "devIdList";
const std::string kProfStart = "prof_start";
const std::string kProfStop = "prof_stop";
const std::string kProfModelSubscribe = "prof_model_subscribe";
const std::string kProfModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kModelName = "model_name";
const std::string kModelId = "model_id";
const std::string kOpNmae = "op_name";
const std::string kOptype = "op_type";
const std::string kBlockDim = "block_dims";
const std::string kTaskId = "task_id";
const std::string kStreamId = "stream_id";
const std::string kThreadId = "thread_id";
const std::string kIndexId = "index_id";
const std::string kTimeStamp = "time_stamp";
const std::string kTagId = "tag_id";
const std::string kShapeType = "shape_type";
const std::string kCurIterNum = "cur_iter_num";
const std::string kTaskType = "task_type";
const std::string kInput = "input";
const std::string kOutput = "output";
const std::string kFormat = "format";
const std::string kDataType = "data_type";
const std::string kShape = "shape";
const std::string kIdx = "idx";

#endif
}  // namespace

namespace ge {
ProfilingManager::ProfilingManager()
    : is_load_profiling_(false),
      is_execute_profiling_(false),
      is_training_trace_(false),
      subscribe_count_(0),
      prof_cb_({nullptr, nullptr}),
      index_id_(UINT64_MAX),
      subscribe_info_({false, 0, 0}) {
}

ProfilingManager::~ProfilingManager() {}

ProfilingManager &ProfilingManager::Instance() {
  static ProfilingManager profiling_manager;
  return profiling_manager;
}

ge::Status ProfilingManager::Init(const Options &options) {
#ifdef DAVINCI_SUPPORT_PROFILING
  vector<int32_t>().swap(device_id_);
  subscribe_count_ = 0;
  GELOGI("ProfilingManager::Init  job_id:%s", options.job_id.c_str());

  struct MsprofGeOptions prof_conf = {{ 0 }};
  Status ret = InitFromOptions(options, prof_conf);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][Profiling]Failed, error_code %u", ret);
    REPORT_CALL_ERROR("E19999", "Init profiling failed, error_code %u", ret);
    return ret;
  }

  if (is_execute_profiling_) {
    if (prof_cb_.msprofCtrlCallback == nullptr) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofCtrlCallback callback is nullptr");
      REPORT_INNER_ERROR("E19999", "MsprofCtrlCallback callback is nullptr");
      return ge::PARAM_INVALID;
    }
    int32_t cb_ret = prof_cb_.msprofCtrlCallback(
        static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS),
        static_cast<void *>(&prof_conf), sizeof(MsprofGeOptions));
    if (cb_ret != 0) {
      GELOGE(FAILED, "[Call][msprofCtrlCallback]Failed, type %u, return %d",
             static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS), cb_ret);
      REPORT_CALL_ERROR("E19999", "Call msprofCtrlCallback failed, type %u, return %d",
                        static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS),
                        cb_ret);
      return FAILED;
    }
    GELOGI("Profiling init success");
  } else {
    GELOGI("The profiling is off, skip the initialization");
  }
#endif
  return SUCCESS;
}

ge::Status ProfilingManager::InitFromOptions(const Options &options, MsprofGeOptions &prof_conf) {
#ifdef DAVINCI_SUPPORT_PROFILING
  // enable profiling by env
  char env_profiling_mode[MMPA_MAX_PATH] = { 0x00 };
  is_execute_profiling_ = false;

  if (options.profiling_mode == "1" && !options.profiling_options.empty()) {
    // enable profiling by ge option
    if (strncpy_s(prof_conf.options, MSPROF_OPTIONS_DEF_LEN_MAX, options.profiling_options.c_str(),
                  MSPROF_OPTIONS_DEF_LEN_MAX - 1) != EOK) {
      GELOGE(INTERNAL_ERROR, "[copy][ProfilingOptions]Failed, options %s",
             options.profiling_options.c_str());
      REPORT_CALL_ERROR("E19999", "Copy profiling_options %s failed",
                        options.profiling_options.c_str());
      return INTERNAL_ERROR;
    }
    is_execute_profiling_ = true;
    GELOGI("The profiling in options is %s, %s. origin option: %s", options.profiling_mode.c_str(), prof_conf.options,
           options.profiling_options.c_str());
  } else {
    (void)mmGetEnv("PROFILING_MODE", env_profiling_mode, MMPA_MAX_PATH);
    (void)mmGetEnv("PROFILING_OPTIONS", prof_conf.options, MSPROF_OPTIONS_DEF_LEN_MAX);
    // The env is invalid
    if ((strcmp("true", env_profiling_mode) != 0) || (strcmp(prof_conf.options, "\0") == 0)) {
      return SUCCESS;
    }
    // enable profiling by env
    is_execute_profiling_ = true;
    GELOGI("The profiling in env is %s, %s", env_profiling_mode, prof_conf.options);
  }

  if (!is_execute_profiling_) {
    return SUCCESS;
  }

  // Parse json str for bp fp
  Status ret = ParseOptions(prof_conf.options);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "[Parse][Options]Parse training trace param %s failed, error_code %u",
           prof_conf.options, ret);
    REPORT_CALL_ERROR("E19999", "Parse training trace param %s failed, error_code %u",
                      prof_conf.options, ret);
    return ge::PARAM_INVALID;
  }

  if (strncpy_s(prof_conf.jobId, MSPROF_OPTIONS_DEF_LEN_MAX, options.job_id.c_str(), MSPROF_OPTIONS_DEF_LEN_MAX - 1) !=
      EOK) {
    GELOGE(INTERNAL_ERROR, "[Copy][JobId]Failed, original job_id %s", options.job_id.c_str());
    REPORT_CALL_ERROR("E19999", "Copy job_id %s failed", options.job_id.c_str());
    return INTERNAL_ERROR;
  }
  GELOGI("Job id: %s, original job id: %s.", prof_conf.jobId, options.job_id.c_str());
#endif
  return ge::SUCCESS;
}

ge::Status ProfilingManager::ParseOptions(const std::string &options) {
  if (options.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]Profiling options is empty");
    REPORT_INNER_ERROR("E19999", "Profiling options is empty");
    return ge::PARAM_INVALID;
  }
  try {
    Json prof_options = Json::parse(options);
    if (options.find(kTrainingTrace) == std::string::npos) {
      return ge::SUCCESS;
    }
    std::string training_trace;
    if (prof_options.contains(kTrainingTrace)) {
      training_trace = prof_options[kTrainingTrace];
    }
    if (training_trace.empty()) {
      GELOGI("Training trace will not take effect.");
      return ge::SUCCESS;
    }
    GELOGI("GE profiling training trace:%s", training_trace.c_str());
    if (training_trace != "on") {
      GELOGE(ge::PARAM_INVALID, "[Check][Param]Training trace param:%s is invalid.",
             training_trace.c_str());
      REPORT_INNER_ERROR("E19999", "Training trace param:%s is invalid.", training_trace.c_str());
      return ge::PARAM_INVALID;
    }
    if (prof_options.contains(kFpPoint)) {
      fp_point_ = prof_options[kFpPoint];
    }
    if (prof_options.contains(kBpPoint)) {
      bp_point_ = prof_options[kBpPoint];
    }
    if (!fp_point_.empty() && !bp_point_.empty()) {
      GELOGI("Training trace bp fp is set, bp_point:%s, fp_point:%s.", bp_point_.c_str(), fp_point_.c_str());
    }
    is_training_trace_ = true;
  } catch (...) {
    GELOGE(FAILED, "[Check][Param]Json prof_conf options is invalid");
    REPORT_INNER_ERROR("E19999", "Json prof_conf options is invalid");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

void ProfilingManager::StopProfiling() {
#ifdef DAVINCI_SUPPORT_PROFILING
  uint64_t module = GetProfilingModule();
  // The following if case will not be executed in normal case, inc case of ProfStopProfiling is abnormal
  int32_t device_num = static_cast<int32_t>(device_id_.size());
  if (device_num != 0) {
    auto device_id_ptr = std::unique_ptr<uint32_t[]>(new (std::nothrow) uint32_t[device_num]);
    if (device_id_ptr == nullptr) {
      GELOGE(FAILED, "[Stop][Profiling]Device id ptr is null.");
      REPORT_INNER_ERROR("E19999", "Stop profiling, device id ptr is null");
      return;
    }
    for (int32_t i = 0; i < device_num; i++) {
      device_id_ptr[i] = static_cast<uint32_t>(device_id_[i]);
    }
    rtError_t rt_ret = rtProfilerStop(module, device_num, device_id_ptr.get());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Call rtProfilerStop failed, ret:%d", rt_ret);
    }
  }

  // stop profiling
  if (prof_cb_.msprofCtrlCallback == nullptr) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofCtrlCallback callback is nullptr");
      REPORT_INNER_ERROR("E19999", "MsprofCtrlCallback callback is nullptr");
      return;
  }
  int32_t cb_ret = prof_cb_.msprofCtrlCallback(static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_FINALIZE),
                                               nullptr, 0);
  if (cb_ret != 0) {
    GELOGW("call msprofCtrlCallback failed, type:%u, return:%d",
           static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_FINALIZE), cb_ret);
    return;
  }
  GELOGI("Stop Profiling success.");
#endif
}

void ProfilingManager::ProfilingOpInputOutInfo(const TaskDescInfo &task, Json &task_json) {
#ifdef DAVINCI_SUPPORT_PROFILING
  for (size_t i = 0; i < task.input_format.size(); i++) {
    Json tmp_input;
    tmp_input[kIdx] = i;
    Format format = task.input_format[i];
    tmp_input[kFormat] = TypeUtils::FormatToSerialString(format);
    DataType data_type = task.input_data_type[i];
    tmp_input[kDataType] = TypeUtils::DataTypeToSerialString(data_type);
    tmp_input[kShape] = task.input_shape[i];
    task_json[kInput] += tmp_input;
  }

  for (size_t i = 0; i < task.output_format.size(); i++) {
    Json tmp_output;
    tmp_output[kIdx] = i;
    Format format = task.output_format[i];
    tmp_output[kFormat] =  TypeUtils::FormatToSerialString(format);
    DataType data_type = task.output_data_type[i];
    tmp_output[kDataType] = TypeUtils::DataTypeToSerialString(data_type);
    tmp_output[kShape] = task.output_shape[i];
    task_json[kOutput] += tmp_output;
  }
#endif
}

void ProfilingManager::ProfilingTaskDescInfo(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info,
                                             const int32_t &device_id) {
#ifdef DAVINCI_SUPPORT_PROFILING
  for (const auto &task : task_desc_info) {
    Json task_info;
    task_info[kModelName] = task.model_name;
    task_info[kModelId] = model_id;
    task_info[kOpNmae] = task.op_name;
    task_info[kOptype] = task.op_type;
    task_info[kBlockDim] = task.block_dim;
    task_info[kTaskType] = task.task_type;
    task_info[kTaskId] = task.task_id;
    task_info[kStreamId] = task.stream_id;
    task_info[kCurIterNum] = task.cur_iter_num;
    task_info[kShapeType] = task.shape_type;
    ProfilingOpInputOutInfo(task, task_info);

    std::string reported_data;
    try {
      reported_data = task_info.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
    } catch (std::exception &e) {
      GELOGE(FAILED, "[Convert][ReportData]Failed to convert json to string, reason %s.",
             e.what());
      REPORT_CALL_ERROR("E19999", "Failed to convert reported_data from json to string, reason %s",
                        e.what());
      return ;
    } catch (...) {
      GELOGE(FAILED, "[Convert][ReportedData]Failed to convert JSON to string");
      REPORT_CALL_ERROR("E19999", "Failed to convert reported data from json to string");
      return;
    }
    reported_data.append(",")
                 .append("\n");
    ReportData(device_id, reported_data, "task_desc_info");
  }
#endif
}

Status ProfilingManager::ProfileStepInfo(uint64_t index_id, uint64_t model_id, uint16_t tag_id, rtStream_t stream,
                                         int32_t device_id) {
#ifdef DAVINCI_SUPPORT_PROFILING
  if (!is_load_profiling_ && subscribe_count_ == 0) {
    GELOGD("Profiling is not turned on, no need to profile step info.");
    return SUCCESS;
  }

  GELOGD("Profiling Step Info TraceTask execute async start, index_id = %lu, model_id = %lu, tag_id = %u",
         index_id, model_id, tag_id);
  rtError_t rt_ret = rtProfilerTraceEx(index_id, model_id, tag_id, stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtProfilerTraceEx]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtProfilerTraceEx failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("Profiling Step Info TraceTask execute async success, index_id = %lu, model_id = %lu, tag_id = %u",
         index_id, model_id, tag_id);

  mmTimespec timespec = mmGetTickCount();
  // 1000 ^ 3 converts second to nanosecond
  int64_t time = timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Get][RtsInfo]Task_id and stream_id failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Get task_id and stream_id failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("Get profiling args, task_id[%u], stream_id[%u]", task_id, stream_id);

  Json step_info;
  step_info[kIndexId] = index_id;
  step_info[kModelId] = model_id;
  step_info[kTimeStamp] = time;
  step_info[kTagId] = tag_id;
  step_info[kTaskId] = task_id;
  step_info[kStreamId] = stream_id;
  step_info[kThreadId] = mmGetTid();

  std::string reported_data;
  try {
    reported_data = step_info.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    GELOGE(FAILED, "Failed to convert JSON to string, reason: %s.", e.what());
    GELOGE(FAILED, "[Convert][ReportedData]Failed to convert from json to string, reason: %s",
           e.what());
    REPORT_CALL_ERROR("E19999", "Failed to convert reported data from json to string, reason: %s",
                      e.what());
  } catch (...) {
    GELOGE(FAILED, "[Convert][ReportedData]Failed to convert from json to string");
    REPORT_CALL_ERROR("E19999", "Failed to convert reported data from json to string");
  }
  reported_data.append(",")
               .append("\n");
  ReportData(device_id, reported_data, "step_info");
#endif
  return SUCCESS;
}

void ProfilingManager::ReportData(const int32_t &device_id, const string &data, const string &tag_name) {
#ifdef DAVINCI_SUPPORT_PROFILING
  ReporterData reporter_data{};
  int ret = -1;
  int32_t cb_ret = -1;
  size_t report_max_len = reporter_max_len_;
  size_t index = data.size() / report_max_len;
  if (index >= 1) {
    reporter_data.deviceId = device_id;
    ret = memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, tag_name.c_str(), tag_name.size());
    GE_IF_BOOL_EXEC(ret != EOK, GELOGE(ret, "Report data tag [%s] memcpy error!", tag_name.c_str()); return;);
    std::lock_guard<std::mutex> lock(mutex_report_);
    for (size_t i = 0; i < index; ++i) {
      reporter_data.data = (unsigned char *)data.c_str() + report_max_len * i;
      reporter_data.dataLen = report_max_len;
      cb_ret = CallMsprofReport(reporter_data);
      GE_IF_BOOL_EXEC(cb_ret != 0, GELOGE(cb_ret, "Reporter data [%s] failed, ret:%d", tag_name.c_str(), cb_ret);
                      return;);
    }
    reporter_data.dataLen = data.size() - report_max_len * index;
    if (reporter_data.dataLen != 0) {
      reporter_data.data = (unsigned char *)data.c_str() + report_max_len * index;
      cb_ret = CallMsprofReport(reporter_data);
      GE_IF_BOOL_EXEC(cb_ret != 0, GELOGE(cb_ret, "Reporter data [%s] failed, ret:%d", tag_name.c_str(), cb_ret);
                      return;);
    }
  } else {
    reporter_data.deviceId = device_id;
    reporter_data.data = (unsigned char *)data.c_str();
    reporter_data.dataLen = data.size();
    ret = memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, tag_name.c_str(), tag_name.size());
    GE_IF_BOOL_EXEC(ret != EOK, GELOGE(ret, "Report data tag [%s] memcpy error!", tag_name.c_str()); return;);
    std::lock_guard<std::mutex> lock(mutex_report_);
    cb_ret = CallMsprofReport(reporter_data);
    GE_IF_BOOL_EXEC(cb_ret != 0, GELOGE(cb_ret, "Reporter data [%s] failed, ret:%d", tag_name.c_str(), cb_ret);
                    return;);
  }
#endif
}

void ProfilingManager::ReportProfilingData(uint32_t model_id, const std::vector<TaskDescInfo> &task_desc_info) {
#ifdef DAVINCI_SUPPORT_PROFILING
  int32_t logic_device_id = 0;
  rtError_t rt_ret = rtGetDevice(&logic_device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Get][LogicDeviceId]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Get logic device id failed, ret 0x%X", rt_ret);
    return;
  }
  GELOGD("current logic_device_id:%d", logic_device_id);
  GELOGD("start ProfilingTaskDescInfo.");
  ProfilingTaskDescInfo(model_id, task_desc_info, logic_device_id);
  GELOGD("Report profiling data for GE end.");
#endif
}

uint64_t ProfilingManager::GetProfilingModule() {
  uint64_t module = PROF_MODEL_EXECUTE_MASK |
                    PROF_RUNTIME_API_MASK |
                    PROF_RUNTIME_TRACE_MASK |
                    PROF_SCHEDULE_TIMELINE_MASK |
                    PROF_SCHEDULE_TRACE_MASK |
                    PROF_TASK_TIME_MASK |
                    PROF_SUBTASK_TIME_MASK |
                    PROF_AICPU_TRACE_MASK |
                    PROF_AICORE_METRICS_MASK |
                    PROF_AIVECTORCORE_METRICS_MASK |
                    PROF_MODEL_LOAD_MASK;
  return module;
}

void ProfilingManager::UpdateSubscribeDeviceModuleMap(std::string prof_type, uint32_t device_id, uint64_t module) {
#ifdef DAVINCI_SUPPORT_PROFILING
  if (prof_type == kProfModelSubscribe) {
    if (subs_dev_module_.find(device_id) != subs_dev_module_.end()) {
      subs_dev_module_[device_id].subscribe_count++;
    } else {
      DeviceSubsInfo dev_info;
      dev_info.module = module;
      dev_info.subscribe_count = 1;
      subs_dev_module_[device_id] = dev_info;
    }
  } else if (prof_type == kProfModelUnsubscribe) {
    auto iter = subs_dev_module_.find(device_id);
    if (iter != subs_dev_module_.end()) {
      if (iter->second.subscribe_count > 0) {
        iter->second.subscribe_count--;
      }
      if (iter->second.subscribe_count == 0) {
        subs_dev_module_.erase(iter);
      }
    }
  } else {
    GELOGI("No need to update device_id module map.");
  }
#endif
}

Status ProfilingManager::ProfModelSubscribe(uint64_t module, void *model) {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t model_load_mask = module & PROF_MODEL_LOAD_MASK;
  if ((subscribe_count_ == 0) && (model_load_mask == PROF_MODEL_LOAD_MASK)) {
    // register framework to profiling
    // register Framework to profiling
    int32_t cb_ret = PluginInit();
    if (cb_ret != 0) {
      GELOGE(cb_ret, "[Init][ProfilingPlugin]Failed, ret %d", cb_ret);
      REPORT_CALL_ERROR("E19999", "Init profiling plugin failed, ret %d", cb_ret);
      return cb_ret;
    }
    GELOGI("Prof subscribe: model load profiling on.");
  }
  subscribe_count_++;

  auto davinci_model = static_cast<DavinciModel *>(model);
  int32_t device_num = 1;
  uint32_t device[1];
  device[0] = davinci_model->GetDeviceId();
  rtError_t rt_ret = rtProfilerStart(module, device_num, device);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Start][Profiler]Malloc buffer failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Malloc buffer failed when start profiling, ret 0x%X", rt_ret);
    return FAILED;
  }
  UpdateSubscribeDeviceModuleMap(kProfModelSubscribe, device[0], module);

  // Report profiling data
  Status p_ret = davinci_model->ReportProfilingData();
  if (p_ret != SUCCESS) {
    GELOGE(p_ret, "[Report][ProfilingData]Failed, ret %u", p_ret);
    REPORT_CALL_ERROR("E19999", "Report profiling data failed, ret %u", p_ret);
    return p_ret;
  }
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfModelUnsubscribe(void *model) {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  if (subscribe_count_ == 0) {
    GELOGW("The profiler has not been subscribed, you do not need to cannel the subscription.");
    return SUCCESS;
  }

  auto davinci_model = static_cast<DavinciModel *>(model);
  int32_t dev_num = 1;
  uint32_t device[1];
  device[0] = davinci_model->GetDeviceId();
  auto iter = subs_dev_module_.find(device[0]);
  if (iter != subs_dev_module_.end()) {
    if (subs_dev_module_[device[0]].subscribe_count == 1) {
      // The same device_id, only stop at last time
      rtError_t rt_ret = rtProfilerStop(subs_dev_module_[device[0]].module, dev_num, device);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(FAILED, "[Stop][Profiler]Malloc buffer Failed, ret %d", rt_ret);
        REPORT_CALL_ERROR("E19999", "Malloc buffer failed when stop profiling, ret %d", rt_ret);
        return FAILED;
      }
    }
    UpdateSubscribeDeviceModuleMap(kProfModelUnsubscribe, device[0], subs_dev_module_[device[0]].module);
  } else {
    GELOGE(FAILED, "[Cancel][DeviceId]The device_id %u has not been subscribed, "
           "do not need to cancel", device[0]);
    REPORT_CALL_ERROR("E19999", "The device_id %u has not been subscribed, do not need to cancel",
                      device[0]);
    return FAILED;
  }

  subscribe_count_--;
  if (subscribe_count_ == 0) {
    // profiling plugin uninit at last subscription
    PluginUnInit();
  }
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfInit(uint64_t module) {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t model_load_mask = module & PROF_MODEL_LOAD_MASK;

  if (model_load_mask == PROF_MODEL_LOAD_MASK) {
    // register Framework to profiling
    int32_t cb_ret = PluginInit();
    if (cb_ret != 0) {
      GELOGE(cb_ret, "[Init][ProfilingPlugin]Failed, ret %d", cb_ret);
      REPORT_CALL_ERROR("E19999", "Init profiling plugin failed, ret %d", cb_ret);
      return cb_ret;
    }

    int32_t device_num = -1;
    rtError_t rt_ret = rtProfilerStart(model_load_mask, device_num, nullptr);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Start][Profiler]Malloc buffer failed, ret 0x%X", rt_ret);
      REPORT_CALL_ERROR("E19999", "Malloc buffer failed when start profiling, ret 0x%X", rt_ret);
      return FAILED;
    }
    is_load_profiling_ = true;
    GELOGI("Prof init: model load profiling on.");
  }

  uint64_t training_trace_mask = module & PROF_TRAINING_TRACE_MASK;
  if (training_trace_mask == PROF_TRAINING_TRACE_MASK) {
    is_training_trace_ = true;
  }
  GELOGI("Prof init success.");
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfFinalize() {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  is_load_profiling_ = false;
  is_training_trace_ = false;
  is_execute_profiling_ = false;
  index_id_ = UINT64_MAX;

  // profiling plugin uninit
  PluginUnInit();

  CleanSubscribeInfo();

  int32_t dev_num = -1;
  rtError_t rt_ret = rtProfilerStop(PROF_MODEL_LOAD_MASK, dev_num, nullptr);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Stop][Profiler]Malloc buffer failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Malloc buffer failed when stop profiling, ret 0x%X", rt_ret);
    return FAILED;
  }
  for (auto device_id_module : device_id_module_map_) {
    if (device_id_module.second != 0) {
      uint32_t device_id = static_cast<uint32_t>(device_id_module.first);
      GELOGI("Prof finalize: device_id: %u, module: 0x%lx.", device_id, device_id_module.second);
      rt_ret = rtProfilerStop(device_id_module.second, 1, &device_id);
      if (rt_ret != RT_ERROR_NONE) {
        GELOGE(FAILED, "[Stop][Profiler]Failed, device_id %d, ret 0x%X", device_id, rt_ret);
        REPORT_CALL_ERROR("E19999", "Stop runtime profiler failed, device_id %d, ret 0x%X",
                          device_id,rt_ret);
        return FAILED;
      }
    }
  }
  device_id_module_map_.clear();
  device_id_.clear();
  device_id_map_.clear();
  model_id_map_.clear();
  GELOGI("Prof finalize success.");
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfParseDeviceId(const std::map<std::string, std::string> &config_para,
                                           vector<int32_t> &device_list) {
#ifdef DAVINCI_SUPPORT_PROFILING
  auto iter = config_para.find(kConfigDevIdList);
  if (iter != config_para.end()) {
    std::string device_id_list = iter->second;
    std::string temp;
    vector<std::string> decvice_id;
    for (uint32_t i = 0; i < device_id_list.size(); i++) {
      if (isdigit(device_id_list[i])) {
        temp.append(1, device_id_list[i]);
      } else {
        if (!temp.empty()) {
          decvice_id.emplace_back(temp);
        }
        temp.clear();
      }
    }
    if (!temp.empty()) {
      decvice_id.emplace_back(temp);
    }

    for (uint32_t i = 0; i < decvice_id.size(); i++) {
      try {
        int32_t dev_id = std::stoi(decvice_id[i]);
        device_list.push_back(dev_id);
      } catch (std::invalid_argument &) {
        GELOGE(FAILED, "[Parse][DeviceId]Failed, it is invalid, %s", decvice_id[i].c_str());
        REPORT_CALL_ERROR("E19999", "Parse device id %s failed, it is invalid",
                          decvice_id[i].c_str());
        return FAILED;
      } catch (std::out_of_range &) {
        GELOGE(FAILED, "[Parse][DeviceId]Failed, it is out of range, %s", decvice_id[i].c_str());
        REPORT_CALL_ERROR("E19999", "Parse device id %s failed, it is out of range",
                          decvice_id[i].c_str());
        return FAILED;
      } catch (...) {
        GELOGE(FAILED, "[Parse][DeviceId]Faield, it cannot change to int, %s",
               decvice_id[i].c_str());
        REPORT_CALL_ERROR("E19999", "Parse device id %s failed, it cannot change to int",
                          decvice_id[i].c_str());
        return FAILED;
      }
    }
  } else {
    GELOGE(FAILED, "[Parse][DeviceId]Config para not contain device id list");
    REPORT_CALL_ERROR("E19999", "Parse device id failed, config para not contain device id list");
    return FAILED;
  }
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                                        vector<int32_t> &device_list) {
#ifdef DAVINCI_SUPPORT_PROFILING
  // device num
  auto iter = config_para.find(kConfigNumsdev);
  if (iter != config_para.end()) {
    try {
      device_num = std::stoi(iter->second);
    } catch (std::invalid_argument &) {
      GELOGE(FAILED, "[Parse][Param]Failed, device num %s is invalid", iter->second.c_str());
      REPORT_CALL_ERROR("E19999", "Parse param failed, device num %s is invalid",
                        iter->second.c_str());
      return FAILED;
    } catch (std::out_of_range &) {
      GELOGE(FAILED, "[Parse][Param]Failed, device num %s cannot change to int",
             iter->second.c_str());
      REPORT_CALL_ERROR("E19999", "Parse param failed, device num %s cannot change to int",
                        iter->second.c_str());
      return FAILED;
    } catch (...) {
      GELOGE(FAILED, "[Parse][Param]Failed, device num %s cannot change to int",
             iter->second.c_str());
      REPORT_CALL_ERROR("E19999", "Parse param failed, device num %s cannot change to int",
                        iter->second.c_str());
      return FAILED;
    }
  } else {
    GELOGE(FAILED, "[Parse][Param]Config para not contain device num %s", iter->second.c_str());
    REPORT_CALL_ERROR("E19999", "Parse param failed, config para not contain device num %s",
                      iter->second.c_str());
    return FAILED;
  }
  // device id
  if (ProfParseDeviceId(config_para, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Parse][DeviceId]Failed");
    REPORT_CALL_ERROR("E19999", "Parse device id failed");
    return FAILED;
  }

  if (device_num == 0 || device_num > kMaxDeviceNum || device_num != static_cast<int32_t>(device_list.size())) {
    GELOGE(FAILED, "[Parse][Param]Failed, config para device num %d not equal to "
           "device list size %zu", device_num, device_list.size());
    REPORT_INNER_ERROR("E19999", "[Parse][Param]Failed, config para device num %d "
                       "not equal to device list size %zu", device_num, device_list.size());
    return FAILED;
  }
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfStartProfiling(uint64_t module, const std::map<std::string, std::string> &config_para) {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t training_trace_mask = module & PROF_TRAINING_TRACE_MASK;
  if (training_trace_mask == PROF_TRAINING_TRACE_MASK) {
    is_training_trace_ = true;
  }
  int32_t device_num = 0;
  vector<int32_t> device_list;
  if (ProfParseParam(config_para, device_num, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Parse][Param]Prof start parse param failed, device num %d, "
           "device list size %zu", device_num, device_list.size());
    REPORT_CALL_ERROR("E19999", "Prof start parse param failed, device num %d, "
                      "device list size %zu", device_num, device_list.size());
    return FAILED;
  }

  auto device_id_ptr = std::unique_ptr<uint32_t[]>(new (std::nothrow) uint32_t[device_num]);
  if (device_id_ptr == nullptr) {
    GELOGE(FAILED, "[Start][Profiling]Malloc buffer failed when start profiling, device num %d",
           device_num);
    REPORT_CALL_ERROR("E19999", "Malloc buffer failed when start profiling, device num %d",
                      device_num);
    return FAILED;
  }
  for (int32_t i = 0; i < device_num; i++) {
    device_id_ptr[i] = static_cast<uint32_t>(device_list[i]);
  }
  GELOGI("Runtime config param: 0x%lx, device num: %d.", module, device_num);

  rtError_t rt_ret = rtProfilerStart(module, device_num, device_id_ptr.get());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Start][Profiler]Runtime profiler config proc failed, config param 0x%lx, "
           "device num %d, ret 0x%X", module, device_num, rt_ret);
    REPORT_CALL_ERROR("E19999", "Runtime profiler config proc failed, config param 0x%lx, "
                      "device num %d, ret 0x%X", module, device_num, rt_ret);
    return FAILED;
  }
  if ((module & PROF_MODEL_EXECUTE_MASK) == PROF_MODEL_EXECUTE_MASK) {
    for (int32_t i = 0; i < device_num; i++) {
      if (std::find(device_id_.begin(), device_id_.end(), device_list[i]) == device_id_.end()) {
        device_id_.push_back(device_list[i]);
      }
    }
    GELOGI("Prof start: ge execute model start profiling.");
  }
  if ((module & PROF_MODEL_LOAD_MASK) == PROF_MODEL_LOAD_MASK) {
    GELOGW("Prof start: load model module is invalid.");
  }
  UpdateDeviceIdModuleMap(kProfStart, module, device_list);
  GELOGI("Prof start profiling success.");
#endif
  return SUCCESS;
}

Status ProfilingManager::ProfStopProfiling(uint64_t module, const std::map<std::string, std::string> &config_para) {
#ifdef DAVINCI_SUPPORT_PROFILING
  std::lock_guard<std::mutex> lock(mutex_);
  int32_t device_num = 0;
  vector<int32_t> device_list;
  if (ProfParseParam(config_para, device_num, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Stop][Profiling]Prof stop parse param failed, device num %d, "
           "device list size %zu", device_num, device_list.size());
    REPORT_CALL_ERROR("E19999", "Prof stop parse param failed, device num %d, device list size %zu",
                      device_num, device_list.size());
    return FAILED;
  }
  auto device_id_ptr = std::unique_ptr<uint32_t[]>(new (std::nothrow) uint32_t[device_num]);
  if (device_id_ptr == nullptr) {
    GELOGE(FAILED, "[Stop][Profiling]Malloc buffer failed when stop profiling, device num %d",
           device_num);
    REPORT_CALL_ERROR("E19999", "Malloc buffer failed when stop profiling, device num %d",
                      device_num);
    return FAILED;
  }
  for (int32_t i = 0; i < device_num; i++) {
    device_id_ptr[i] = static_cast<uint32_t>(device_list[i]);
  }
  GELOGI("Prof stop: runtime config param: 0x%lx, device num: %d", module, device_num);
  rtError_t rt_ret = rtProfilerStop(module, device_num, device_id_ptr.get());
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Stop][Profiler]Runtime profiler config proc failed, config param 0x%lx, "
           "device num: %d, ret 0x%X", module, device_num, rt_ret);
    REPORT_CALL_ERROR("E19999", "Runtime profiler config proc failed, config param 0x%lx, "
                      "device num %d, ret 0x%X", module, device_num, rt_ret);
    return FAILED;
  }
  uint64_t execute_model_mask = module & PROF_MODEL_EXECUTE_MASK;
  if (execute_model_mask == PROF_MODEL_EXECUTE_MASK) {
    for (int32_t i = 0; i < device_num; i++) {
      auto iter = std::find(device_id_.begin(), device_id_.end(), device_list[i]);
      if (iter != device_id_.end()) {
        device_id_.erase(iter);
      }
    }
    GELOGI("Prof stop: ge execute model stop profiling.");
  }
  if ((module & PROF_MODEL_LOAD_MASK) == PROF_MODEL_LOAD_MASK) {
    GELOGW("Prof stop: load model module is invalid.");
  }
  UpdateDeviceIdModuleMap(kProfStop, module, device_list);
  GELOGI("Prof stop profiling success.");
#endif
  return SUCCESS;
}

void ProfilingManager::UpdateDeviceIdModuleMap(string prof_type, uint64_t module, const vector<int32_t> &device_list) {
#ifdef DAVINCI_SUPPORT_PROFILING
  if (prof_type == kProfStart) {
    for (uint32_t i = 0; i < device_list.size(); i++) {
      auto iter = device_id_module_map_.find(device_list[i]);
      if (iter != device_id_module_map_.end()) {
        uint64_t prof_on_module = device_id_module_map_[device_list[i]];
        // save all profiling on module of device
        device_id_module_map_[device_list[i]] = prof_on_module | module;
      } else {
        device_id_module_map_[device_list[i]] = module;
      }
    }
  } else if (prof_type == kProfStop) {
    for (uint32_t i = 0; i < device_list.size(); i++) {
      auto iter = device_id_module_map_.find(device_list[i]);
      if (iter != device_id_module_map_.end()) {
        uint64_t prof_on_module = device_id_module_map_[device_list[i]];
        uint64_t prof_off_module = prof_on_module & module;
        uint64_t prof_on_left_module = prof_on_module & (~prof_off_module);
        // stop profiling on module of device
        device_id_module_map_[device_list[i]] = prof_on_left_module;
      }
    }
  } else {
    GELOGI("No need to update device_id module map.");
  }
#endif
}

bool ProfilingManager::ProfilingModelExecuteOn() const {
  int32_t logic_device_id = 0;
  rtError_t rt_ret = rtGetDevice(&logic_device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Get][LogicDeviceId]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Get logic device id failed, ret 0x%X", rt_ret);
  }
  GELOGI("Current logic_device_id:%d", logic_device_id);

  bool execute_model_prof_on = false;
  auto iter = std::find(device_id_.begin(), device_id_.end(), logic_device_id);
  if (iter != device_id_.end()) {
    execute_model_prof_on = true;
  }
  GELOGI("Flag is_execute_profiling: %d, execute_model_prof_on: %d", is_execute_profiling_, execute_model_prof_on);
  return  execute_model_prof_on;
}

Status ProfilingManager::PluginInit() {
  if (prof_cb_.msprofReporterCallback == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofReporterCallback callback is nullptr");
    REPORT_INNER_ERROR("E19999", "MsprofReporterCallback callback is nullptr");
    return ge::PARAM_INVALID;
  }
  int32_t cb_ret = prof_cb_.msprofReporterCallback(
      static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
      static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_INIT),
      nullptr, 0);
  if (cb_ret != MSPROF_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Profiling reporter init failed, ret 0x%X", cb_ret);
    GELOGE(INTERNAL_ERROR, "[Init][ProfilingReporter]Failed, ret 0x%X", cb_ret);
    return INTERNAL_ERROR;
  }

  cb_ret = prof_cb_.msprofReporterCallback(
      static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
      static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_DATA_MAX_LEN),
      &reporter_max_len_, sizeof(uint32_t));
  if (cb_ret != MSPROF_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Get profiling reporter data max len failed, ret 0x%X", cb_ret);
    GELOGE(INTERNAL_ERROR, "[Get][ProfilingDataMaxLen]Failed, ret 0x%X", cb_ret);
    return INTERNAL_ERROR;
  }

 return SUCCESS;
}

void ProfilingManager::PluginUnInit() const {
#ifdef DAVINCI_SUPPORT_PROFILING
  if (prof_cb_.msprofReporterCallback == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofReporterCallback callback is nullptr");
    REPORT_INNER_ERROR("E19999", "MsprofReporterCallback callback is nullptr");
    return;
  }
  int32_t cb_ret = prof_cb_.msprofReporterCallback(
      static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
      static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_UNINIT),
      nullptr, 0);
  if (cb_ret != 0) {
    GELOGW("profiling plugin uninit failed, ret:%d", cb_ret);
  }
#endif
}

Status ProfilingManager::CallMsprofReport(ReporterData &reporter_data) const {
  if (prof_cb_.msprofReporterCallback == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofReporterCallback callback is nullptr");
    REPORT_INNER_ERROR("E19999", "MsprofReporterCallback callback is nullptr");
    return ge::PARAM_INVALID;
  }
  return prof_cb_.msprofReporterCallback(
      static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
      static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_REPORT),
      static_cast<void *>(&reporter_data), sizeof(ReporterData));
}

void ProfilingManager::GetOpInputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  std::vector<Format> input_format;
  std::vector<std::vector<int64_t>> input_shape;
  std::vector<DataType> input_data_type;
  for (size_t i = 0; i < op->GetAllInputsSize(); ++i) {
    GeTensorDescPtr input_tensor_desc = op->MutableInputDesc(i);
    if (input_tensor_desc == nullptr) {
      continue;
    }
    input_format.emplace_back(input_tensor_desc->GetFormat());
    input_shape.emplace_back(input_tensor_desc->GetShape().GetDims());
    input_data_type.emplace_back(input_tensor_desc->GetDataType());
  }

  std::vector<Format> format_default =  { FORMAT_NULL };
  std::vector<std::vector<int64_t>> shape_default = { {0} };
  std::vector<DataType> data_type_default = { DT_UNDEFINED };
  task_desc_info.input_format = input_format.empty() ? format_default : input_format;
  task_desc_info.input_shape = input_shape.empty() ? shape_default : input_shape;
  task_desc_info.input_data_type = input_data_type.empty() ? data_type_default : input_data_type;
}

void ProfilingManager::GetOpOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  std::vector<Format> output_format;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<DataType> output_data_type;
  for (size_t j = 0; j < op->GetOutputsSize(); ++j) {
    GeTensorDescPtr output_tensor_desc = op->MutableOutputDesc(j);
    if (output_tensor_desc == nullptr) {
      continue;
    }
    output_format.emplace_back(output_tensor_desc->GetFormat());
    output_shape.emplace_back(output_tensor_desc->GetShape().GetDims());
    output_data_type.emplace_back(output_tensor_desc->GetDataType());
  }

  std::vector<Format> format_default =  { FORMAT_NULL };
  std::vector<std::vector<int64_t>> shape_default = { {0} };
  std::vector<DataType> data_type_default = { DT_UNDEFINED };
  task_desc_info.output_format = output_format.empty() ? format_default : output_format;
  task_desc_info.output_shape = output_shape.empty() ? shape_default : output_shape;
  task_desc_info.output_data_type = output_data_type.empty() ? data_type_default : output_data_type;
}

void ProfilingManager::GetOpInputOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  GetOpInputInfo(op, task_desc_info);
  GetOpOutputInfo(op, task_desc_info);
}

void ProfilingManager::GetFpBpPoint(std::string &fp_point, std::string &bp_point) {
  // Env or options mode, fp_point_/bp_point_ have initiliazed on profiling init
  if (!fp_point_.empty() && !bp_point_.empty()) {
    fp_point = fp_point_;
    bp_point = bp_point_;
    GELOGI("Bp Fp have been initialized in env or options. bp_point: %s, fp_point: %s", bp_point.c_str(),
           fp_point.c_str());
    return;
  }
  // ProfApi mode and training trace is set
  // Parse options first
  char env_profiling_options[MSPROF_OPTIONS_DEF_LEN_MAX] = {0x00};
  bool is_profiling_valid = false;
  std::string profiling_options;
  if (ge::GetContext().GetOption(OPTION_EXEC_PROFILING_OPTIONS, profiling_options) == SUCCESS &&
      !profiling_options.empty()) {
    is_profiling_valid = true;
  } else {
    INT32 ret = mmGetEnv("PROFILING_OPTIONS", env_profiling_options, MSPROF_OPTIONS_DEF_LEN_MAX);
    if (ret != EN_OK) {
      GELOGI("PROFILING_OPTIONS env is not exist.");
      return;
    }
    GELOGI("Parse env PROFILING_OPTIONS:%s.", env_profiling_options);
    profiling_options = env_profiling_options;
    is_profiling_valid = true;
  }
  if (is_profiling_valid) {
    try {
      Json prof_options = Json::parse(profiling_options);
      if (prof_options.contains(kFpPoint)) {
        fp_point_ = prof_options[kFpPoint];
      }
      if (prof_options.contains(kBpPoint)) {
        bp_point_ = prof_options[kBpPoint];
      }
      fp_point = fp_point_;
      bp_point = bp_point_;
      if (!fp_point_.empty() && !bp_point_.empty()) {
        GELOGI("Training trace bp fp is set, bp_point:%s, fp_point:%s.", bp_point_.c_str(), fp_point_.c_str());
      }
    } catch (...) {
      GELOGW("Json prof options is invalid.");
      return;
    }
  }

  return;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ProfilingManager::GetDeviceIdFromGraph(
    uint32_t graph_id, uint32_t &device_id) {
  auto iter = device_id_map_.find(graph_id);
  if (iter != device_id_map_.end()) {
    device_id = iter->second;
    return SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "graph_id:%u does not exist!", graph_id);
  GELOGE(PARAM_INVALID, "[Check][GraphId]graph_id:%u does not exist!", graph_id);
  return FAILED;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void ProfilingManager::SetSubscribeInfo(
    uint64_t prof_switch, uint32_t model_id, bool is_subscribe) {
  subscribe_info_.is_subscribe = is_subscribe;
  subscribe_info_.prof_switch = prof_switch;
  subscribe_info_.graph_id = model_id;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void ProfilingManager::CleanSubscribeInfo() {
  subscribe_info_.is_subscribe = false;
  subscribe_info_.prof_switch = 0;
  subscribe_info_.graph_id = 0;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ProfilingManager::GetModelIdFromGraph(
    uint32_t graph_id, uint32_t &model_id) {
  auto iter = model_id_map_.find(graph_id);
  if (iter != model_id_map_.end()) {
    model_id = iter->second;
    return SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "graph_id:%u does not exist!", graph_id);
  GELOGE(PARAM_INVALID, "[Check][GraphId]graph_id:%u does not exist!", graph_id);
  return FAILED;
}
}  // namespace ge
