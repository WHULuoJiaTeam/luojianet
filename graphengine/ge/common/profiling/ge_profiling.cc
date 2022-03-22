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

#include "framework/common/profiling/ge_profiling.h"
#include "runtime/base.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/load/graph_loader.h"
#include "graph/ge_context.h"
#include "init/gelib.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/model/ge_model.h"
#include "framework/omg/omg_inner_types.h"

namespace {
const uint32_t kDeviceListIndex = 3;
const std::string kDeviceNums = "devNums";
const std::string kDeviceIdList = "devIdList";
const std::string kProfilingInit = "prof_init";
const std::string kProfilingFinalize = "prof_finalize";
const std::string kProfilingStart = "prof_start";
const std::string kProfilingStop = "prof_stop";
const std::string kProfModelSubscribe = "prof_model_subscribe";
const std::string kProfModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kRtSetDeviceRegName = "profiling";
const std::string kPofilingModelId = "modelId";

const std::map<ProfCommandHandleType, std::string> kProfCommandTypeMap = {
    {kProfCommandhandleInit, kProfilingInit},
    {kProfCommandhandleStart, kProfilingStart},
    {kProfCommandhandleStop, kProfilingStop},
    {kProfCommandhandleFinalize, kProfilingFinalize},
    {kProfCommandhandleModelSubscribe, kProfModelSubscribe},
    {kProfCommandhandleModelUnsubscribe, kProfModelUnsubscribe}};

const uint64_t kModelId = ge::INVALID_MODEL_ID;
const uint16_t kStepStart = 0;
const uint16_t kStepEnd = 1;

ge::Status NeedUnsubscribe(ProfCommandHandleType type, bool is_subscribe,
                           uint32_t graph_id, vector<string> &prof_params) {
  if (type == kProfCommandhandleModelUnsubscribe && is_subscribe) {
    prof_params.clear();
    prof_params.emplace_back(kPofilingModelId);
    uint32_t model_id = 0;
    auto ret = ge::ProfilingManager::Instance().GetModelIdFromGraph(graph_id, model_id);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "graph_id:%u not not found", graph_id);
      return ret;
    }
    prof_params.emplace_back(std::to_string(model_id));
  }
  return ge::SUCCESS;
}
}  // namespace

bool TransProfConfigToParam(const ProfCommandHandleData &profCommand, vector<string> &prof_config_params) {
  prof_config_params.clear();
  prof_config_params.emplace_back(kDeviceNums);
  prof_config_params.emplace_back(std::to_string(profCommand.devNums));
  prof_config_params.emplace_back(kDeviceIdList);
  std::string devID = "";
  if (profCommand.devNums == 0) {
    GELOGW("The device num is invalid.");
    return false;
  }
  for (uint32_t i = 0; i < profCommand.devNums; i++) {
    devID.append(std::to_string(profCommand.devIdList[i]));
    if (i != profCommand.devNums - 1) {
      devID.append(",");
    }
  }

  prof_config_params.push_back(devID);
  return true;
}

bool isProfConfigValid(const uint32_t *deviceid_list, uint32_t device_nums) {
  if (deviceid_list == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][DeviceIDList]Invalid, it is nullptr");
    REPORT_INNER_ERROR("E19999", "Device id list is nullptr");
    return false;
  }
  if (device_nums == 0 || device_nums > MAX_DEV_NUM) {
    GELOGE(ge::PARAM_INVALID, "[Check][DeviceNums]Invalid, device nums: %u", device_nums);
    REPORT_INNER_ERROR("E19999", "DeviceNums %u check invalid", device_nums);
    return false;
  }

  // real device num
  int32_t dev_count = 0;
  rtError_t rt_err = rtGetDeviceCount(&dev_count);
  if (rt_err != RT_ERROR_NONE) {
    GELOGE(ge::INTERNAL_ERROR, "[Get][DeviceCount]Failed, error_code %d", rt_err);
    REPORT_CALL_ERROR("E19999", "Get device count failed, error_code %d", rt_err);
    return false;
  }

  if (device_nums > static_cast<uint32_t>(dev_count)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]Device num %u is not in range [1,%d]",
           device_nums, dev_count);
    REPORT_INNER_ERROR("E19999", "Device num %u check invalid, it is not in range [1,%d]",
                       device_nums, dev_count);
    return false;
  }

  std::set<uint32_t> record;
  for (size_t i = 0; i < device_nums; ++i) {
    uint32_t dev_id = deviceid_list[i];
    if (dev_id >= static_cast<uint32_t>(dev_count)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DeviceId]Device id %u is not in range [0,%d)",
             dev_id, dev_count);
      REPORT_CALL_ERROR("E19999", "Device id %u is not in range [0,%d)", dev_id, dev_count);
      return false;
    }
    if (record.count(dev_id) > 0) {
      GELOGE(ge::PARAM_INVALID, "[Check][DeviceId]Device id %u is duplicatedly set", dev_id);
      REPORT_CALL_ERROR("E19999", "Device id %u is not unique, duplicatedly set", dev_id);
      return false;
    }
    record.insert(dev_id);
  }
  return true;
}

ge::Status RegProfCtrlCallback(MsprofCtrlCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]Msprof ctrl callback is nullptr");
    REPORT_INNER_ERROR("E19999", "Msprof ctrl callback is nullptr");
    return ge::PARAM_INVALID;
  }
  if (ge::ProfilingManager::Instance().GetMsprofCallback().msprofCtrlCallback != nullptr) {
    GELOGW("Msprof ctrl callback is exist, just ignore it.");
  } else {
    ge::ProfilingManager::Instance().SetMsprofCtrlCallback(func);
  }
  return ge::SUCCESS;
}

ge::Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofSetDeviceCallback callback is nullptr");
    REPORT_INNER_ERROR("E19999", "MsprofSetDeviceCallback callback is nullptr");
    return ge::PARAM_INVALID;
  }
  // Pass MsprofSetDeviceCallback to runtime
  ge::Status rt_ret = rtRegDeviceStateCallback(kRtSetDeviceRegName.c_str(), static_cast<rtDeviceStateCallback>(func));
  if (rt_ret != ge::SUCCESS) {
    GELOGE(rt_ret, "[Pass][MsprofSetDeviceCallback]To runtime failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Pass MsprofSetDeviceCallback to runtime failed, ret 0x%X", rt_ret);
    return rt_ret;
  }
  return ge::SUCCESS;
}

ge::Status RegProfReporterCallback(MsprofReporterCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]MsprofReporterCallback callback is nullptr");
    REPORT_INNER_ERROR("E19999", "MsprofReporterCallback callback is nullptr");
    return ge::PARAM_INVALID;
  }
  if (ge::ProfilingManager::Instance().GetMsprofCallback().msprofReporterCallback != nullptr) {
    GELOGW("Msprof reporter callback is exist, just ignore it.");
  } else {
    GELOGI("GE register Msprof reporter callback.");
    ge::ProfilingManager::Instance().SetMsprofReporterCallback(func);
    // Pass MsprofReporterCallback to runtime
    ge::Status rt_ret = rtSetMsprofReporterCallback(func);
    if (rt_ret != ge::SUCCESS) {
      GELOGE(rt_ret, "[Pass][Param]Pass MsprofReporterCallback to runtime failed, error_code %u",
             rt_ret);
      REPORT_CALL_ERROR("E19999", "Pass MsprofReporterCallback to runtime failed, error_code %u",
                        rt_ret);
      return rt_ret;
    }
    // Pass MsprofReporterCallback to hccl
  }
  return ge::SUCCESS;
}

ge::Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len) {
  if (type != kProfCommandhandleFinalize) {
    GE_CHECK_NOTNULL(data);
  }
  ProfCommandHandleData *prof_config_param = reinterpret_cast<ProfCommandHandleData *>(data);
  auto iter = kProfCommandTypeMap.find(type);
  if (iter == kProfCommandTypeMap.end()) {
    GELOGW("The prof comand type is invalid.");
    return ge::PARAM_INVALID;
  }
  std::vector<string> prof_params;
  if (type == kProfCommandhandleStart || type == kProfCommandhandleStop) {
    if (!isProfConfigValid(prof_config_param->devIdList, prof_config_param->devNums)) {
      return ge::FAILED;
    }

    if (!TransProfConfigToParam(*prof_config_param, prof_params)) {
      GELOGE(ge::PARAM_INVALID, "[Check][Param]Transfer profilerConfig to string vector failed");
      REPORT_CALL_ERROR("E19999", "Transfer profilerConfig to string vector failed");
      return ge::PARAM_INVALID;
    }
  }
  auto &profiling_manager = ge::ProfilingManager::Instance();
  auto is_train = domi::GetContext().train_flag;
  if (type == kProfCommandhandleModelSubscribe && is_train) {
    profiling_manager.SetSubscribeInfo(prof_config_param->profSwitch, prof_config_param->modelId, true);
    return ge::SUCCESS;
  }
  auto is_subscribe = profiling_manager.GetSubscribeInfo().is_subscribe;
  // GraphId is actually stored in prof_config_param
  auto graph_id = prof_config_param->modelId;
  ge::Status ret = NeedUnsubscribe(type, is_subscribe, graph_id, prof_params);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "graph_id:%u not not found", graph_id);
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"value", "parameter", "reason"}),
                       std::vector<std::string>({std::to_string(graph_id),
                                                 "GraphToModelMap",
                                                 "graph_id does not exist!"}));
    return ge::FAILED;
  }
  ge::GraphLoader graph_loader;
  ge::Command command;
  command.cmd_params.clear();
  command.cmd_type = iter->second;
  command.cmd_params = prof_params;
  if (type != kProfCommandhandleFinalize) {
    command.module_index = prof_config_param->profSwitch;
  }
  GELOGI("GE commandhandle execute, Command Type: %s, data type config: 0x%lx", iter->second.c_str(),
         command.module_index);
  if (type == kProfCommandhandleStart || type == kProfCommandhandleStop) {
    GELOGI("Profiling device nums:%s , deviceID:[%s]", prof_params[0].c_str(), prof_params[kDeviceListIndex].c_str());
  }
  ret = graph_loader.CommandHandle(command);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "[Handle][Command]Handle profiling command failed, command type %s, error_code %u",
           iter->second.c_str(), ret);
    REPORT_CALL_ERROR("E19999", "Handle profiling command failed, command type %s, error_code %u",
                      iter->second.c_str(), ret);
    return ge::FAILED;
  }

  GELOGI("Successfully execute profiling command type: %d, command 0x%lx.", type, command.module_index);
  return ge::SUCCESS;
}

ge::Status ProfSetStepInfo(uint64_t index_id, uint16_t tag_id, rtStream_t stream) {
  static bool is_first_run = true;
  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Get][LogicDeviceId]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Get logic device id failed, ret 0x%X", rt_ret);
    return ge::FAILED;
  }
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetStepInfoIndex(index_id);
  if (is_first_run && tag_id == kStepStart) {
    GE_CHK_STATUS_RET_NOLOG(profiling_manager.ProfileStepInfo(index_id, kModelId, tag_id, stream, device_id));
    is_first_run = false;
    return ge::SUCCESS;
  }
  if (!is_first_run && tag_id == kStepEnd) {
    GE_CHK_STATUS_RET_NOLOG(profiling_manager.ProfileStepInfo(index_id, kModelId, tag_id, stream, device_id));
    is_first_run = true;
    return ge::SUCCESS;
  }
  GELOGE(ge::FAILED, "Param tag_id:%u invalid when is_first_run is %d", tag_id, is_first_run);
  REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"value", "parameter", "reason"}),
                     std::vector<std::string>({std::to_string(tag_id), "tag_id",
                                               "tag id must be 0 when first run, must be 1 when second run"}));
  return ge::FAILED;
}

ge::Status ProfGetDeviceFormGraphId(uint32_t graph_id, uint32_t &device_id) {
  return ge::ProfilingManager::Instance().GetDeviceIdFromGraph(graph_id, device_id);
}
