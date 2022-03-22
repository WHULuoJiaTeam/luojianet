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

#include "common/ge/op_tiling_manager.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/log.h"
#include <string>

namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const std::string kDefaultPath = "/usr/local/Ascend/opp";
const std::string kDefaultBuiltInTilingPath = "/op_impl/built-in/ai_core/tbe/op_tiling/liboptiling.so";
const std::string kDefaultCustomTilingPath = "/op_impl/custom/ai_core/tbe/op_tiling/liboptiling.so";
const uint8_t kPrefixIndex = 9;
}  // namespace

namespace ge {
void OpTilingManager::ClearHandles() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGE(FAILED, "[Close][Handle]Failed, handle of %s: %s", handle.first.c_str(), error);
      REPORT_CALL_ERROR("E19999", "Failed to close handle of %s: %s",
                        handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

OpTilingManager::~OpTilingManager() { ClearHandles(); }

std::string OpTilingManager::GetPath() {
  char opp_path_env[MMPA_MAX_PATH] = { 0x00 };
  INT32 res = mmGetEnv(kEnvName, opp_path_env, MMPA_MAX_PATH);
  std::string opp_path = kDefaultPath;
  if (res == EN_OK) {
    char resolved_path[MMPA_MAX_PATH];
    if (mmRealPath(opp_path_env, resolved_path, MMPA_MAX_PATH) != EN_OK) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E19024", {"env", "value", "situation"}, {"ASCEND_OPP_PATH", opp_path_env, "loading the tiling lib"});
      GELOGE(PARAM_INVALID, "[Load][TilingLib]Failed, as env 'ASCEND_OPP_PATH'[%s] "
             "is invalid path. errmsg:%s", opp_path_env, strerror(errno));
      return std::string();
    }
    opp_path = resolved_path;
  }
  return opp_path;
}

void OpTilingManager::LoadSo() {
  std::string opp_path = GetPath();
  if (opp_path.empty()) {
    GELOGW("Skip load tiling lib.");
    return;
  }
  std::string built_in_tiling_lib = opp_path + kDefaultBuiltInTilingPath;
  std::string custom_tiling_lib = opp_path + kDefaultCustomTilingPath;
  std::string built_in_name = kDefaultBuiltInTilingPath.substr(kPrefixIndex);
  std::string custom_name = kDefaultCustomTilingPath.substr(kPrefixIndex);

  void *handle_bi = mmDlopen(built_in_tiling_lib.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
  if (handle_bi == nullptr) {
    const char *error = mmDlerror();
    GE_IF_BOOL_EXEC(error == nullptr, error = "");
    GELOGW("Failed to dlopen %s! errmsg:%s", built_in_tiling_lib.c_str(), error);
  } else {
    handles_[built_in_name] = handle_bi;
  }

  void *handle_ct = mmDlopen(custom_tiling_lib.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
  if (handle_ct == nullptr) {
    const char *error = mmDlerror();
    GE_IF_BOOL_EXEC(error == nullptr, error = "");
    GELOGW("Failed to dlopen %s! errmsg:%s", custom_tiling_lib.c_str(), error);
  } else {
    handles_[custom_name] = handle_ct;
  }
}

OpTilingManager &OpTilingManager::GetInstance() {
  static OpTilingManager instance;
  return instance;
}
}  // namespace ge
