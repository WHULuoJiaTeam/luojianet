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
#ifndef LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_

#include <map>
#include <cstring>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include "include/common/utils/contract.h"
#include "utils/ms_context.h"
#include "toolchain/prof_callback.h"
#include "toolchain/prof_acl_api.h"
#include "toolchain/slog.h"
#include "runtime/base.h"
#include "profiler/device/profiling.h"
#include "acl/acl_prof.h"

using std::map;
using std::string;

namespace luojianet_ms {
namespace device {
namespace ascend {
struct MsprofCallback {
  MsprofCtrlCallback msprofCtrlCallback;
  MsprofSetDeviceCallback msprofSetDeviceCallback;
  MsprofReporterCallback msprofReporterCallback;
};

enum ProfCommandHandleType {
  kProfCommandhandleInit = 0,
  kProfCommandhandleStart,
  kProfCommandhandleStop,
  kProfCommandhandleFinalize,
  kProfCommandhandleModelSubscribe,
  kProfCommandhandleModelUnsubscribe
};

enum ProfilingState { kProfilingInvalid, kProfilingInit, kProfilingStart, kProfilingStop, kProfilingFinalize };

class ProfilingManager {
 public:
  static ProfilingManager &GetInstance();
  uint64_t GetJobId() const;
  bool ProfRegisterCtrlCallback() const;
  bool InitProfiling(const std::string &profiling_path, uint32_t device_id);
  bool IsProfilingInitialized() const { return cur_state_ >= kProfilingInit; }
  inline bool IsProfilingStart() const { return cur_state_ >= kProfilingStart; }
  Status PluginInit() const;
  void PluginUnInit() const;
  Status CallMsprofReport(NotNull<ReporterData *> reporter_data) const;
  void QueryHashId(const int32_t &device_id, const std::string &src_str, uint64_t *hash_id);
  const struct MsprofCallback &GetMsprofCallback() { return prof_cb_; }
  void SetMsprofCtrlCallback(MsprofCtrlCallback func) { prof_cb_.msprofCtrlCallback = func; }
  void SetMsprofReporterCallback(MsprofReporterCallback func) { prof_cb_.msprofReporterCallback = func; }
  void SetMsprofSetDeviceCallback(MsprofSetDeviceCallback func) { prof_cb_.msprofSetDeviceCallback = func; }
  Status GetProfConf(NotNull<MsprofGeOptions *> prof);
  Status ProfCommandHandle(ProfCommandHandleType type);
  Status ProfHandleInit();
  Status ProfHandleStart();
  Status ProfHandleStop();
  Status ProfHandleFinalize();
  void ReportErrorMessage() const;

 protected:
  ProfilingManager();
  ~ProfilingManager() {}

 private:
  uint32_t device_id_;
  MsprofCallback prof_cb_;
  ProfilingState cur_state_;
  std::string profiling_path_;
};

Status ProfCommandHandle(ProfCommandHandleType type);
rtError_t CtrlCallbackHandle(uint32_t rt_type, void *data, uint32_t /*len*/);
Status ProfCtrlSwitchHandle(void *data);
}  // namespace ascend
}  // namespace device
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
