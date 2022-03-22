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

#include "common/dump/dump_manager.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

namespace {
const char *const kDumpOFF = "OFF";
const char *const kDumpoff = "off";
const char *const kDumpOn = "on";
const uint64_t kInferSessionId = 0;
const uint32_t kAllOverflow = 3;
}  // namespace
namespace ge {
DumpManager &DumpManager::GetInstance() {
  static DumpManager instance;
  return instance;
}

bool DumpManager::NeedDoDump(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (dump_config.dump_status.empty() && dump_config.dump_debug.empty()) {
    dump_properties_map_[kInferSessionId] = dump_properties;
    GELOGI("Dump does not open");
    return false;
  }
  GELOGI("Dump status is %s, dump debug is %s.", dump_config.dump_status.c_str(), dump_config.dump_debug.c_str());
  if ((dump_config.dump_status == kDumpoff || dump_config.dump_status == kDumpOFF) &&
       dump_config.dump_debug == kDumpoff) {
    dump_properties.ClearDumpPropertyValue();
    dump_properties_map_[kInferSessionId] = dump_properties;
    return false;
  }
  if (dump_config.dump_status == kDumpOn && dump_config.dump_debug == kDumpOn) {
    GELOGW("Not support coexistence of dump debug and dump status.");
    return false;
  }
  return true;
}

void DumpManager::SetDumpDebugConf(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (dump_config.dump_debug == kDumpOn) {
    GELOGI("Only do overflow detection, dump debug is %s.", dump_config.dump_debug.c_str());
    dump_properties.InitInferOpDebug();
    dump_properties.SetOpDebugMode(kAllOverflow);
  }
}

void DumpManager::SetDumpList(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  for (const auto &model_dump : dump_config.dump_list) {
    std::string model_name = model_dump.model_name;
    GELOGI("Dump model is %s", model_name.c_str());
    std::set<std::string> dump_layers;
    for (const auto &layer : model_dump.layers) {
      GELOGI("Dump layer is %s in model", layer.c_str());
      dump_layers.insert(layer);
    }
    dump_properties.AddPropertyValue(model_name, dump_layers);
  }
}

Status DumpManager::SetNormalDumpConf(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (dump_config.dump_status == kDumpOn) {
    GELOGI("Only do normal dump process, dump status is %s", dump_config.dump_status.c_str());
    dump_properties.SetDumpStatus(dump_config.dump_status);
    std::string dump_op_switch = dump_config.dump_op_switch;
    dump_properties.SetDumpOpSwitch(dump_op_switch);
    if (dump_op_switch == kDumpoff && dump_config.dump_list.empty()) {
      dump_properties_map_.emplace(kInferSessionId, dump_properties);
      GELOGE(PARAM_INVALID, "[Check][DumpList]Invalid, dump_op_switch is %s", dump_op_switch.c_str());
      REPORT_INNER_ERROR("E19999", "Dump list check invalid, dump_op_switch is %s", dump_op_switch.c_str());
      return PARAM_INVALID;
    }

    if (!dump_config.dump_list.empty()) {
      if (dump_op_switch == kDumpOn) {
        GELOGI("Start to dump model and single op, dump op switch is %s", dump_op_switch.c_str());
      } else {
        GELOGI("Only dump model, dump op switch is %s", dump_op_switch.c_str());
      }
      SetDumpList(dump_config, dump_properties);
    } else {
      GELOGI("Only dump single op, dump op switch is %s", dump_op_switch.c_str());
    }
    GELOGI("Dump mode is %s", dump_config.dump_mode.c_str());
    dump_properties.SetDumpMode(dump_config.dump_mode);
  }
  return SUCCESS;
}

Status DumpManager::SetDumpPath(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  std::string dump_path = dump_config.dump_path;
  if (dump_path.empty()) {
    GELOGE(PARAM_INVALID, "[Check][DumpPath]It is empty.");
    REPORT_INNER_ERROR("E19999", "Dump path check is empty.");
    return PARAM_INVALID;
  }
  if (dump_path[dump_path.size() - 1] != '/') {
    dump_path = dump_path + "/";
  }
  dump_path = dump_path + CurrentTimeInStr() + "/";
  GELOGI("Dump path is %s", dump_path.c_str());
  dump_properties.SetDumpPath(dump_path);
  return SUCCESS;
}

Status DumpManager::SetDumpConf(const DumpConfig &dump_config) {
  DumpProperties dump_properties;
  if (!NeedDoDump(dump_config, dump_properties)) {
    GELOGD("No need do dump process.");
    return SUCCESS;
  }
  SetDumpDebugConf(dump_config, dump_properties);
  GE_CHK_STATUS_RET(SetNormalDumpConf(dump_config, dump_properties), "[Init][DumpConf] failed when dump status is on.");
  GE_CHK_STATUS_RET(SetDumpPath(dump_config, dump_properties), "[Init][DumpPath] failed.");
  dump_properties_map_[kInferSessionId] = dump_properties;
  
  return SUCCESS;
}

const DumpProperties &DumpManager::GetDumpProperties(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    return iter->second;
  }
  static DumpProperties default_properties;
  return default_properties;
}

void DumpManager::AddDumpProperties(uint64_t session_id, const DumpProperties &dump_properties) {
  std::lock_guard<std::mutex> lock(mutex_);
  dump_properties_map_.emplace(session_id, dump_properties);
}

void DumpManager::RemoveDumpProperties(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    dump_properties_map_.erase(iter);
  }
}

}  // namespace ge
