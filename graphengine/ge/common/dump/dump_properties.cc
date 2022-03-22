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

#include "common/dump/dump_properties.h"

#include <cstdio>
#include <string>
#include <regex>

#include "common/ge/ge_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"

namespace {
const std::string kEnableFlag = "1";
const std::string kDumpStatusOpen = "on";
const uint32_t kAicoreOverflow = (0x1 << 0);
const uint32_t kAtomicOverflow = (0x1 << 1);
const uint32_t kAllOverflow = (kAicoreOverflow | kAtomicOverflow);
}  // namespace
namespace ge {
void DumpProperties::Split(const std::string &s, std::vector<std::string> &result, const char *delchar) {
  if (s.empty()) {
    return;
  }
  result.clear();

  char *buffer = new (std::nothrow)char[s.size() + 1];
  if (buffer == nullptr) {
    GELOGE(FAILED, "[Split][string] failed while malloc memory, string value is:%s", s.c_str());
    REPORT_CALL_ERROR("E19999", "Memory malloc may fail when split string, get fatal exception, "
                      "string value is:%s", s.c_str());
    return;
  }
  buffer[s.size()] = '\0';
  errno_t e = strcpy_s(buffer, s.size() + 1, s.c_str());
  if (e != EOK) {
    delete[] buffer;
    return;
  }
  char *context = nullptr;
  char *p = strtok_s(buffer, delchar, &context);
  while (p != nullptr) {
    result.emplace_back(p);
    p = strtok_s(nullptr, delchar, &context);
  }
  delete[] buffer;
}

Status DumpProperties::CheckDumpStep(const std::string &dump_step) {
  std::string modified_dum_step = dump_step + "|";
  std::smatch result;
  std::vector<string> match_vecs;
  std::regex pattern(R"((\d{1,}-\d{1,}\||\d{1,}\|)+)");
  if (regex_match(modified_dum_step, result, pattern)) {
    Split(result.str(), match_vecs, "|");
    if (match_vecs.empty()) {
      REPORT_CALL_ERROR("E19999", "Split may get fatal exception, dump_step:%s.", dump_step.c_str());
      GELOGE(FAILED, "[Check][Param] failed. Split may get fatal exception, ge.exec.dumpStep:%s.", dump_step.c_str());
      return FAILED;
    }
    // 100 is the max sets of dump steps.
    if (match_vecs.size() > 100) {
      REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({
                         "ge.exec.dumpStep",
                         dump_step.c_str(),
                         " is not supported, only support dump <= 100 sets of data"}));
      GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
             "dump_step only support dump <= 100 sets of data.", dump_step.c_str());
      return PARAM_INVALID;
    }
    for (const auto &match_vec : match_vecs) {
      std::vector<string> vec_after_split;
      Split(match_vec, vec_after_split, "-");
      if (match_vecs.empty()) {
        REPORT_CALL_ERROR("E19999", "Split may get fatal exception.");
        GELOGE(FAILED, "[Check][Param] failed, split may get fatal exception.");
        return FAILED;
      }
      if (vec_after_split.size() > 1) {
        if (std::atoi(vec_after_split[0].c_str()) >= std::atoi(vec_after_split[1].c_str())) {
          REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                             std::vector<std::string>({
                             "ge.exec.dumpStep",
                             dump_step.c_str(),
                             " is not supported."
                             "in range steps, the first step is >= second step, correct example:'0|5|10-20"}));
          GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
          "in range steps, the first step is >= second step, correct example:'0|5|10-20'", dump_step.c_str());
          return PARAM_INVALID;
        }
      }
    }
  } else {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.dumpStep",
                       dump_step.c_str(),
                       " is not supported, correct example:'0|5|10|50-100."}));
    GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
    "dump_step string style is error, correct example:'0|5|10|50-100.'", dump_step.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckDumpMode(const std::string &dump_mode) {
  const std::set<string> dump_mode_list = {"input", "output", "all"};
  std::set<string>::iterator iter;

  if ((iter = dump_mode_list.find(dump_mode)) == dump_mode_list.end()) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.dumpMode",
                       dump_mode.c_str(),
                       " is not supported, should be one of the following:[input, output, all]"}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dump_debug_mode:%s, is is not supported,"
           "should be one of the following:[input, output, all].", dump_mode.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckDumpPath(const std::string &input) {
  if (mmIsDir(input.c_str()) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.dumpPath",
                       input.c_str(),
                       " is not a directory."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, is not directory.", input.c_str());
    return PARAM_INVALID;
  }
  char trusted_path[MMPA_MAX_PATH] = { "\0" };
  if (mmRealPath(input.c_str(), trusted_path, MMPA_MAX_PATH) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.dumpPath",
                       input.c_str(),
                       " dumpPath invalid."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dumpPath:%s, is invalid.", input.c_str());
    return PARAM_INVALID;
  }
  if (mmAccess2(trusted_path, M_R_OK | M_W_OK) != EN_OK) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.dumpPath",
                       input.c_str(),
                       " does't have read, write permissions."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, does't have read, write permissions.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckEnableDump(const std::string &input) {
  std::set<string> enable_dump_option_list = {"1", "0"};
  auto it = enable_dump_option_list.find(input);
  if (it == enable_dump_option_list.end()) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.enableDump",
                       input.c_str(),
                       " only support 1 or 0."}));
    GELOGE(PARAM_INVALID, "[Check][Param] Not support ge.exec.enableDump or ge.exec.enableDumpDebug format:%s, "
           "only support 1 or 0.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

DumpProperties::DumpProperties(const DumpProperties &other) {
  CopyFrom(other);
}

DumpProperties &DumpProperties::operator=(const DumpProperties &other) {
  CopyFrom(other);
  return *this;
}

Status DumpProperties::SetDumpOptions() {
  if (enable_dump_ == kEnableFlag) {
    std::string dump_step;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_STEP, dump_step) == GRAPH_SUCCESS && !dump_step.empty()) {
      GE_CHK_STATUS_RET(CheckDumpStep(dump_step), "[Check][dump_step] failed.");
      GELOGI("Get dump step %s successfully", dump_step.c_str());
      SetDumpStep(dump_step);
    }
    string dump_mode = "output";
    if (GetContext().GetOption(OPTION_EXEC_DUMP_MODE, dump_mode) == GRAPH_SUCCESS) {
      GELOGI("Get dump mode %s successfully", dump_mode.c_str());
      GE_CHK_STATUS_RET(CheckDumpMode(dump_mode), "[Check][dump_mode] failed.");
      SetDumpMode(dump_mode);
    }
    AddPropertyValue(DUMP_ALL_MODEL, {});
  }
  return SUCCESS;
}

Status DumpProperties::InitByOptions() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  is_train_op_debug_ = false;
  is_infer_op_debug_ = false;
  op_debug_mode_ = 0;

  std::string enable_dump = std::to_string(false);
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP, enable_dump);
  enable_dump_ = enable_dump;
  if (!enable_dump_.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump_), "[Check][enable_dump] failed.");
  }

  std::string enable_dump_debug = std::to_string(false);
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP_DEBUG, enable_dump_debug);
  enable_dump_debug_ = enable_dump_debug;
  if (!enable_dump_debug_.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump_debug_), "[Check][enable_dump_debug] failed.");
  }
  if ((enable_dump_ == kEnableFlag) && (enable_dump_debug_ == kEnableFlag)) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({
                       "ge.exec.enableDump and ge.exec.enableDumpDebug",
                       enable_dump_ + ", " + enable_dump_debug,
                       "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be set to 1 at the same time."}));
    GELOGE(FAILED, "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be both set to 1 at the same time.");
    return FAILED;
  }
  if ((enable_dump_ == kEnableFlag) || (enable_dump_debug_ == kEnableFlag)) {
    std::string dump_path;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_PATH, dump_path) == GRAPH_SUCCESS) {
      GE_CHK_STATUS_RET(CheckDumpPath(dump_path), "Check dump path failed.");
      if (!dump_path.empty() && dump_path[dump_path.size() - 1] != '/') {
        dump_path = dump_path + "/";
      }
      dump_path = dump_path + CurrentTimeInStr() + "/";
      GELOGI("Get dump path %s successfully", dump_path.c_str());
      SetDumpPath(dump_path);
    } else {
      REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({
                         "ge.exec.dumpPath",
                         dump_path,
                         "ge.exec.dumpPath is not set."}));
      GELOGE(FAILED, "[Check][dump_path] failed. Dump path is not set.");
      return FAILED;
    }
  }

  GE_CHK_STATUS_RET(SetDumpOptions(), "SetDumpOptions failed.");

  GE_CHK_STATUS_RET(SetDumpDebugOptions(), "SetDumpDebugOptions failed.");

  return SUCCESS;
}

// The following is the new dump scenario of the fusion operator
void DumpProperties::AddPropertyValue(const std::string &model, const std::set<std::string> &layers) {
  for (const std::string &layer : layers) {
    GELOGI("This model %s config to dump layer %s", model.c_str(), layer.c_str());
  }

  model_dump_properties_map_[model] = layers;
}

void DumpProperties::DeletePropertyValue(const std::string &model) {
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    model_dump_properties_map_.erase(iter);
  }
}

void DumpProperties::ClearDumpPropertyValue() {
  model_dump_properties_map_.clear();
}

void DumpProperties::ClearDumpInfo() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  dump_op_switch_.clear();
  dump_status_.clear();
  is_train_op_debug_ = false;
  is_infer_op_debug_ = false;
  op_debug_mode_ = 0;
}

std::set<std::string> DumpProperties::GetAllDumpModel() const {
  std::set<std::string> model_list;
  for (auto &iter : model_dump_properties_map_) {
    model_list.insert(iter.first);
  }

  return model_list;
}

std::set<std::string> DumpProperties::GetPropertyValue(const std::string &model) const {
  auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return iter->second;
  }
  return {};
}

bool DumpProperties::IsLayerNeedDump(const std::string &model, const std::string &om_name,
                                     const std::string &op_name) const {
  // if dump all
  GELOGD("model name is %s om name is %s op is %s in layer need dump", model.c_str(), om_name.c_str(), op_name.c_str());
  if (model_dump_properties_map_.find(DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }

  // if this model need dump
  auto om_name_iter = model_dump_properties_map_.find(om_name);
  auto model_name_iter = model_dump_properties_map_.find(model);
  if (om_name_iter != model_dump_properties_map_.end() || model_name_iter != model_dump_properties_map_.end()) {
    // if no dump layer info, dump all layer in this model
    auto model_iter = om_name_iter != model_dump_properties_map_.end() ? om_name_iter : model_name_iter;
    if (model_iter->second.empty()) {
      return true;
    }

    return model_iter->second.find(op_name) != model_iter->second.end();
  }

  GELOGD("Model %s is not seated to be dump", model.c_str());
  return false;
}

void DumpProperties::SetDumpPath(const std::string &path) {
  dump_path_ = path;
}

const std::string &DumpProperties::GetDumpPath() const {
  return dump_path_;
}

void DumpProperties::SetDumpStep(const std::string &step) {
  dump_step_ = step;
}

const std::string &DumpProperties::GetDumpStep() const {
  return dump_step_;
}

void DumpProperties::SetDumpMode(const std::string &mode) {
  dump_mode_ = mode;
}

const std::string &DumpProperties::GetDumpMode() const {
  return dump_mode_;
}

void DumpProperties::SetDumpStatus(const std::string &status) {
  dump_status_ = status;
}

const std::string &DumpProperties::GetDumpStatus() const {
  return dump_status_;
}

void DumpProperties::InitInferOpDebug() {
  is_infer_op_debug_ = true;
}

void DumpProperties::SetOpDebugMode(const uint32_t &op_debug_mode) {
  op_debug_mode_ = op_debug_mode;
}

void DumpProperties::SetDumpOpSwitch(const std::string &dump_op_switch) {
  dump_op_switch_ = dump_op_switch;
}

const std::string &DumpProperties::GetDumpOpSwitch() const {
  return dump_op_switch_;
}

bool DumpProperties::IsSingleOpNeedDump() const {
  if (dump_op_switch_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

bool DumpProperties::IsDumpOpen() const {
  if (enable_dump_ == kEnableFlag || dump_status_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

void DumpProperties::CopyFrom(const DumpProperties &other) {
  if (&other != this) {
    enable_dump_ = other.enable_dump_;
    enable_dump_debug_ = other.enable_dump_debug_;
    dump_path_ = other.dump_path_;
    dump_step_ = other.dump_step_;
    dump_mode_ = other.dump_mode_;
    dump_status_ = other.dump_status_;
    dump_op_switch_ = other.dump_op_switch_;

    model_dump_properties_map_ = other.model_dump_properties_map_;
    is_train_op_debug_ = other.is_train_op_debug_;
    is_infer_op_debug_ = other.is_infer_op_debug_;
    op_debug_mode_ = other.op_debug_mode_;
  }
}

Status DumpProperties::SetDumpDebugOptions() {
  if (enable_dump_debug_ == kEnableFlag) {
    std::string dump_debug_mode;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DEBUG_MODE, dump_debug_mode) == GRAPH_SUCCESS) {
      GELOGD("Get ge.exec.dumpDebugMode %s successfully.", dump_debug_mode.c_str());
    } else {
      GELOGW("ge.exec.dumpDebugMode is not set.");
      return SUCCESS;
    }

    if (dump_debug_mode == OP_DEBUG_AICORE) {
      GELOGD("ge.exec.dumpDebugMode=aicore_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAicoreOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ATOMIC) {
      GELOGD("ge.exec.dumpDebugMode=atomic_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAtomicOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ALL) {
      GELOGD("ge.exec.dumpDebugMode=all, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAllOverflow;
    } else {
      REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                         std::vector<std::string>({
                         "ge.exec.dumpDebugMode",
                         dump_debug_mode,
                         "ge.exec.dumpDebugMode is invalid."}));
      GELOGE(PARAM_INVALID, "[Set][DumpDebugOptions] failed, ge.exec.dumpDebugMode is invalid.");
      return PARAM_INVALID;
    }
  } else {
    GELOGI("ge.exec.enableDumpDebug is false or is not set");
  }
  return SUCCESS;
}
}  // namespace ge
