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

#include "init/gelib.h"

#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"
#include "framework/omg/ge_init.h"
#include "analyzer/analyzer.h"
#include "external/ge/ge_api_types.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "common/ge_call_wrapper.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/kernel.h"
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "external/runtime/rt_error_codes.h"

using Json = nlohmann::json;

namespace ge {
namespace {
const int kDecimal = 10;
const int kSocVersionLen = 50;
const int kDefaultDeviceIdForTrain = 0;
const int kDefaultDeviceIdForInfer = -1;
const char *const kGlobalOptionFpCeilingModeDefault = "2";
}  // namespace
static std::shared_ptr<GELib> instancePtr_ = nullptr;

// Initial each module of GE, if one failed, release all
Status GELib::Initialize(const map<string, string> &options) {
  GELOGI("initial start");
  GEEVENT("[GEPERFTRACE] GE Init Start");
  // Multiple initializations are not allowed
  instancePtr_ = MakeShared<GELib>();
  if (instancePtr_ == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "[Create][GELib]GeLib initialize failed, malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "GELib Init failed for new GeLib failed.");
    return GE_CLI_INIT_FAILED;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kSystemInit);
  map<string, string> new_options;
  Status ret = instancePtr_->SetRTSocVersion(options, new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][RTSocVersion]GeLib initial: SetRTSocVersion failed.");
    REPORT_CALL_ERROR("E19999", "SetRTSocVersion failed.");
    return ret;
  }

  ret = instancePtr_->SetAiCoreNum(new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][AiCoreNum]GeLib initial: SetAiCoreNum failed.");
    REPORT_CALL_ERROR("E19999", "SetAiCoreNum failed.");
    return ret;
  }

  instancePtr_->SetDefaultPrecisionMode(new_options);

  if (new_options.find("ge.fpCeilingMode") == new_options.end()) {
    new_options["ge.fpCeilingMode"] = kGlobalOptionFpCeilingModeDefault;
  }

  GetMutableGlobalOptions().insert(new_options.begin(), new_options.end());
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GE_TIMESTAMP_START(Init);
  ret = instancePtr_->InnerInitialize(new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GeLib]GeLib initial failed.");
    REPORT_CALL_ERROR("E19999", "GELib::InnerInitialize failed.");
    instancePtr_ = nullptr;
    return ret;
  }
  GE_TIMESTAMP_EVENT_END(Init, "GELib::Initialize");
  return SUCCESS;
}

Status GELib::InnerInitialize(const map<string, string> &options) {
  // Multiple initializations are not allowed
  if (init_flag_) {
    GELOGW("multi initializations");
    return SUCCESS;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kSystemInit);
  GELOGI("GE System initial.");
  GE_TIMESTAMP_START(SystemInitialize);
  Status initSystemStatus = SystemInitialize(options);
  GE_TIMESTAMP_END(SystemInitialize, "InnerInitialize::SystemInitialize");
  if (initSystemStatus != SUCCESS) {
    GELOGE(initSystemStatus, "[Init][GESystem]GE system initial failed.");
    RollbackInit();
    return initSystemStatus;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kEngineInit);
  GELOGI("engineManager initial.");
  GE_TIMESTAMP_START(EngineInitialize);
  Status initEmStatus = engineManager_.Initialize(options);
  GE_TIMESTAMP_END(EngineInitialize, "InnerInitialize::EngineInitialize");
  if (initEmStatus != SUCCESS) {
    GELOGE(initEmStatus, "[Init][EngineManager]GE engine manager initial failed.");
    REPORT_CALL_ERROR("E19999", "EngineManager initialize failed.");
    RollbackInit();
    return initEmStatus;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kOpsKernelInit);
  GELOGI("opsManager initial.");
  GE_TIMESTAMP_START(OpsManagerInitialize);
  Status initOpsStatus = opsManager_.Initialize(options);
  GE_TIMESTAMP_END(OpsManagerInitialize, "InnerInitialize::OpsManagerInitialize");
  if (initOpsStatus != SUCCESS) {
    GELOGE(initOpsStatus, "[Init][OpsManager]GE ops manager initial failed.");
    REPORT_CALL_ERROR("E19999", "OpsManager initialize failed.");
    RollbackInit();
    return initOpsStatus;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kOpsKernelBuilderInit);
  GELOGI("opsBuilderManager initial.");
  GE_TIMESTAMP_START(OpsKernelBuilderManagerInitialize);
  Status initOpsBuilderStatus = OpsKernelBuilderManager::Instance().Initialize(options);
  GE_TIMESTAMP_END(OpsKernelBuilderManagerInitialize, "InnerInitialize::OpsKernelBuilderManager");
  if (initOpsBuilderStatus != SUCCESS) {
    GELOGE(initOpsBuilderStatus, "[Init][OpsKernelBuilderManager]GE ops builder manager initial failed.");
    REPORT_CALL_ERROR("E19999", "OpsBuilderManager initialize failed.");
    RollbackInit();
    return initOpsBuilderStatus;
  }

  GELOGI("Start to initialize HostCpuEngine");
  GE_TIMESTAMP_START(HostCpuEngineInitialize);
  Status initHostCpuEngineStatus = HostCpuEngine::GetInstance().Initialize();
  GE_TIMESTAMP_END(HostCpuEngineInitialize, "InnerInitialize::HostCpuEngineInitialize");
  if (initHostCpuEngineStatus != SUCCESS) {
    GELOGE(initHostCpuEngineStatus, "[Init][HostCpuEngine]Failed to initialize HostCpuEngine.");
    REPORT_CALL_ERROR("E19999", "HostCpuEngine initialize failed.");
    RollbackInit();
    return initHostCpuEngineStatus;
  }

  GELOGI("Start to init Analyzer!");
  Status init_analyzer_status = ge::Analyzer::GetInstance()->Initialize();
  if (init_analyzer_status != SUCCESS) {
    GELOGE(init_analyzer_status, "[Init][Analyzer]Failed to initialize Analyzer.");
    REPORT_CALL_ERROR("E19999", "ge::Analyzer initialize failed.");
    RollbackInit();
    return init_analyzer_status;
  }

  init_flag_ = true;
  return SUCCESS;
}

Status GELib::SystemInitialize(const map<string, string> &options) {
  Status status = FAILED;
  auto iter = options.find(OPTION_GRAPH_RUN_MODE);
  if (iter != options.end()) {
    if (GraphRunMode(std::strtol(iter->second.c_str(), nullptr, kDecimal)) >= TRAIN) {
      is_train_mode_ = true;
    }
  }

  InitOptions(options);

  // In train and infer, profiling is always needed.
  InitProfiling(this->options_);
  // 1.`is_train_mode_` means case: train
  // 2.`(!is_train_mode_) && (options_.device_id != kDefaultDeviceIdForInfer)` means case: online infer
  // these two case with logical device id
  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    status = InitSystemWithOptions(this->options_);
  } else {
    status = InitSystemWithoutOptions();
  }
  return status;
}

void GELib::InitProfiling(Options &options) {
  GELOGI("Init Profiling. session Id: %ld, device id:%d ", options.session_id, options.device_id);
  std::lock_guard<std::mutex> lock(status_mutex_);
  GetContext().Init();
  // Profiling init
  if (ProfilingManager::Instance().Init(options) != SUCCESS) {
    GELOGW("Profiling init failed.");
  }
}

void GELib::SetDefaultPrecisionMode(map<string, string> &new_options) {
  auto iter = new_options.find(PRECISION_MODE);
  if (iter != new_options.end()) {
    GELOGI("Find precision_mode in options, value is %s", iter->second.c_str());
    return;
  }
  iter = new_options.find(OPTION_GRAPH_RUN_MODE);
  if (iter != new_options.end()) {
    if (GraphRunMode(std::strtol(iter->second.c_str(), nullptr, kDecimal)) >= TRAIN) {
      // only train mode need to be set allow_fp32_to_fp16.
      GELOGI("This is train mode, precision_mode need to be set allow_fp32_to_fp16");
      new_options.insert(std::make_pair(PRECISION_MODE, "allow_fp32_to_fp16"));
      return;
    }
  }
  GELOGI("This is not train mode, precision_mode need to be set force_fp16");
  new_options.insert(std::make_pair(PRECISION_MODE, "force_fp16"));
  return;
}

Status GELib::SetRTSocVersion(const map<string, string> &options, map<string, string> &new_options) {
  GELOGI("Start to set SOC_VERSION");
  new_options.insert(options.begin(), options.end());
  auto it = new_options.find(ge::SOC_VERSION);
  if (it != new_options.end()) {
    GE_CHK_RT_RET(rtSetSocVersion(it->second.c_str()));
    GELOGI("Succeeded in setting SOC_VERSION[%s] to runtime.", it->second.c_str());
  } else {
    GELOGI("SOC_VERSION is not exist in options");
    char version[kSocVersionLen] = {0};
    rtError_t rt_ret = rtGetSocVersion(version, kSocVersionLen);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, 
        REPORT_CALL_ERROR("E19999", "rtGetSocVersion failed.");
        GELOGE(rt_ret, "[Get][SocVersion]rtGetSocVersion failed");
        return FAILED;)
    GELOGI("Succeeded in getting SOC_VERSION[%s] from runtime.", version);
    new_options.insert(std::make_pair(ge::SOC_VERSION, version));
  }
  return SUCCESS;
}

Status GELib::SetAiCoreNum(map<string, string> &options) {
  // Already set or get AICORE_NUM from options in offline mode
  if (options.find(AICORE_NUM) != options.end()) {
    return SUCCESS;
  }

  uint32_t aicore_num = 0;
  rtError_t ret = rtGetAiCoreCount(&aicore_num);
  if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {  // offline without ATC Input of AiCoreNum
    return SUCCESS;
  } else if (ret == RT_ERROR_NONE) {  // online-mode
    options.emplace(std::make_pair(AICORE_NUM, std::to_string(aicore_num)));
    return SUCCESS;
  }
  GELOGE(FAILED, "[Get][AiCoreCount]rtGetAiCoreCount failed.");
  REPORT_CALL_ERROR("E19999", "rtGetAiCoreCount failed.");
  return FAILED;
}

void GELib::InitOptions(const map<string, string> &options) {
  this->options_.session_id = 0;
  auto iter = options.find(OPTION_EXEC_SESSION_ID);
  if (iter != options.end()) {
    this->options_.session_id = std::strtoll(iter->second.c_str(), nullptr, kDecimal);
  }
  this->options_.device_id = is_train_mode_ ? kDefaultDeviceIdForTrain : kDefaultDeviceIdForInfer;
  iter = options.find(OPTION_EXEC_DEVICE_ID);
  if (iter != options.end()) {
    this->options_.device_id = static_cast<int32_t>(std::strtol(iter->second.c_str(), nullptr, kDecimal));
  }
  iter = options.find(OPTION_EXEC_JOB_ID);
  if (iter != options.end()) {
    this->options_.job_id = iter->second.c_str();
  }
  this->options_.isUseHcom = false;
  iter = options.find(OPTION_EXEC_IS_USEHCOM);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.isUseHcom;
  }
  this->options_.isUseHvd = false;
  iter = options.find(OPTION_EXEC_IS_USEHVD);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.isUseHvd;
  }
  this->options_.deployMode = false;
  iter = options.find(OPTION_EXEC_DEPLOY_MODE);
  if (iter != options.end()) {
    std::istringstream(iter->second) >> this->options_.deployMode;
  }
  iter = options.find(OPTION_EXEC_POD_NAME);
  if (iter != options.end()) {
    this->options_.podName = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_PROFILING_MODE);
  if (iter != options.end()) {
    this->options_.profiling_mode = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_PROFILING_OPTIONS);
  if (iter != options.end()) {
    this->options_.profiling_options = iter->second.c_str();
  }
  iter = options.find(OPTION_EXEC_RANK_ID);
  if (iter != options.end()) {
    this->options_.rankId = std::strtoll(iter->second.c_str(), nullptr, kDecimal);
  }
  iter = options.find(OPTION_EXEC_RANK_TABLE_FILE);
  if (iter != options.end()) {
    this->options_.rankTableFile = iter->second.c_str();
  }
  this->options_.enable_atomic = true;
  iter = options.find(OPTION_EXEC_ATOMIC_FLAG);
  GE_IF_BOOL_EXEC(iter != options.end(),
                  this->options_.enable_atomic = std::strtol(iter->second.c_str(), nullptr, kDecimal));
  GELOGI("ge InnerInitialize, the enable_atomic_flag in options_ is %d", this->options_.enable_atomic);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status GELib::InitSystemWithOptions(Options &options) {
  std::string mode = is_train_mode_ ? "Training" : "Online infer";
  GELOGI("%s init GELib. session Id:%ld, device id :%d ", mode.c_str(), options.session_id, options.device_id);
  GEEVENT("System init with options begin, job id %s", options.job_id.c_str());
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_system_inited && !is_shutdown,
                  GELOGW("System init with options is already inited and not shutdown.");
                  return SUCCESS);

  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  mem_type.push_back(RT_MEMORY_P2P_DDR);
  Status initMmStatus = MemManager::Instance().Initialize(mem_type);
  if (initMmStatus != SUCCESS) {
    GELOGE(initMmStatus, "[Init][MemManager] MemoryAllocatorManager initialize failed.");
    REPORT_CALL_ERROR("E19999", "MemManager initialize failed.");
    return initMmStatus;
  }

  GE_CHK_STATUS_RET(HostMemManager::Instance().Initialize());

  // set device id
  GELOGI("set logical device id:%u", options.device_id);
  GetContext().SetCtxDeviceId(static_cast<uint32_t>(options.device_id));
  GE_CHK_RT_RET(rtSetDevice(options.device_id));

  // In the scenario that the automatic add fusion is set, but there is no cleanaddr operator,
  // maybe need to check it
  is_system_inited = true;
  is_shutdown = false;

  GELOGI("%s init GELib success.", mode.c_str());

  return SUCCESS;
}

Status GELib::SystemShutdownWithOptions(const Options &options) {
  std::string mode = is_train_mode_ ? "Training" : "Online infer";
  GELOGI("%s finalize GELib begin.", mode.c_str());
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_shutdown || !is_system_inited,
                  GELOGW("System Shutdown with options is already is_shutdown or system does not inited. "
                         "is_shutdown:%d is_omm_inited:%d",
                         is_shutdown, is_system_inited);
                  return SUCCESS);

  GE_CHK_RT(rtDeviceReset(options.device_id));

  is_system_inited = false;
  is_shutdown = true;
  GELOGI("%s finalize GELib success.", mode.c_str());
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status GELib::InitSystemWithoutOptions() {
  GELOGI("Inference Init GELib begin.");

  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  mem_type.push_back(RT_MEMORY_P2P_DDR);
  Status initMmStatus = MemManager::Instance().Initialize(mem_type);
  if (initMmStatus != SUCCESS) {
    GELOGE(initMmStatus, "[Init][MemoryManager] initialize failed.");
    REPORT_CALL_ERROR("E19999", "MemManager initialize failed.");
    return initMmStatus;
  }
  GE_CHK_STATUS_RET(HostMemManager::Instance().Initialize());

  static bool is_inited = false;
  if (is_inited) {
    GELOGW("System init without options is already inited,  don't need to init again.");
    return SUCCESS;
  }
  is_inited = true;
  GELOGI("Inference init GELib success.");

  return SUCCESS;
}

string GELib::GetPath() { return PluginManager::GetPath(); }

// Finalize all modules
Status GELib::Finalize() {
  ErrorManager::GetInstance().SetStage(error_message::kFinalize, error_message::kFinalize);
  GELOGI("finalization start");
  // Finalization is not allowed before initialization
  if (!init_flag_) {
    GELOGW("not initialize");
    return SUCCESS;
  }
  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    GE_CHK_RT_RET(rtSetDevice(options_.device_id));
  }
  Status final_state = SUCCESS;
  Status mid_state;
  GELOGI("engineManager finalization.");
  mid_state = engineManager_.Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("engineManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("opsBuilderManager finalization.");
  mid_state = OpsKernelBuilderManager::Instance().Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("opsBuilderManager finalize failed");
    final_state = mid_state;
  }
  GELOGI("opsManager finalization.");
  mid_state = opsManager_.Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("opsManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("VarManagerPool finalization.");
  VarManagerPool::Instance().Destory();

  GELOGI("MemManager finalization.");
  MemManager::Instance().Finalize();

  GELOGI("HostMemManager finalization.");
  HostMemManager::Instance().Finalize();

  GELOGI("HostCpuEngine finalization.");
  HostCpuEngine::GetInstance().Finalize();

  GELOGI("Analyzer finalization");
  Analyzer::GetInstance()->Finalize();

  // Shut down profiling
  ShutDownProfiling();

  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    GELOGI("System ShutDown.");
    mid_state = SystemShutdownWithOptions(this->options_);
    if (mid_state != SUCCESS) {
      GELOGW("System shutdown with options failed");
      final_state = mid_state;
    }
  }

  is_train_mode_ = false;

  GetMutableGlobalOptions().erase(ENABLE_SINGLE_STREAM);

  if (is_train_mode_ || (options_.device_id != kDefaultDeviceIdForInfer)) {
    GE_CHK_RT_RET(rtDeviceReset(options_.device_id));
  }

  instancePtr_ = nullptr;
  init_flag_ = false;
  if (final_state != SUCCESS) {
    GELOGE(FAILED, "[Check][State]finalization failed.");
    REPORT_INNER_ERROR("E19999", "GELib::Finalize failed.");
    return final_state;
  }
  GELOGI("finalization success.");
  return SUCCESS;
}

void GELib::ShutDownProfiling() {
  std::lock_guard<std::mutex> lock(status_mutex_);

  if (ProfilingManager::Instance().ProfilingOn()) {
    ProfilingManager::Instance().StopProfiling();
    ProfilingManager::Instance().PluginUnInit();
  }
}

// Get Singleton Instance
std::shared_ptr<GELib> GELib::GetInstance() { return instancePtr_; }

void GELib::RollbackInit() {
  if (engineManager_.init_flag_) {
    (void)engineManager_.Finalize();
  }
  if (opsManager_.init_flag_) {
    (void)opsManager_.Finalize();
  }
  MemManager::Instance().Finalize();
  HostMemManager::Instance().Finalize();
  VarManagerPool::Instance().Destory();
}

Status GEInit::Initialize(const map<string, string> &options) {
  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    ret = GELib::Initialize(options);
  }
  return ret;
}

Status GEInit::Finalize() {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr != nullptr) {
    return instance_ptr->Finalize();
  }
  return SUCCESS;
}

string GEInit::GetPath() {
  return GELib::GetPath();
}
}  // namespace ge
