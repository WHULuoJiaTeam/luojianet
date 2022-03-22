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

#include "engine_manager/dnnengine_manager.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <utility>

#include "framework/common/debug/log.h"
#include "common/ge/ge_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "analyzer/analyzer.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "init/gelib.h"
#include "framework/common/types.h"

namespace {
const char *const kSchedulerUnits = "schedule_units";
const char *const kId = "id";
const char *const kName = "name";
const char *const kExAttrs = "ex_attrs";
const char *const kIndependent = "independent";
const char *const kSkipAssignStream = "skip_assign_stream";
const char *const kCalEngines = "cal_engines";
const char *const kAttch = "attach";
const char *const kVectorCore = "VectorCore";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kHostCpuEngineName = "DNN_VM_HOST_CPU";
const char *const kHostCpuOpKernelLibName = "DNN_VM_HOST_CPU_OP_STORE";
}  // namespace

namespace ge {
namespace {
const std::set<std::string> kNotCpuOp = {DATA, CONSTANT, CONSTANTOP, VARIABLE, NETOUTPUT};

bool ExecOnHostCpu(const OpDescPtr &op_desc) {
  bool is_host_cpu_op = (kNotCpuOp.find(op_desc->GetType()) == kNotCpuOp.end());
  return ge::GetContext().GetHostExecFlag() && is_host_cpu_op;
}
}  // namespace

DNNEngineManager::DNNEngineManager() : init_flag_(false) {}
DNNEngineManager::~DNNEngineManager() {
  engines_attrs_map_.clear();
  schedulers_.clear();
}

Status DNNEngineManager::Initialize(const std::map<std::string, std::string> &options) {
  // Multiple initializations are not supported
  if (init_flag_) {
    GELOGW("DNNEngineManager has been initialized.");
    return SUCCESS;
  }

  // Load engine so
  std::string so_path = "plugin/nnengine/";
  std::string path = PluginManager::GetPath();
  path.append(so_path);
  std::string so_api_func = "GetDNNEngineObjs";
  std::vector<std::string> so_func{so_api_func};
  Status status = plugin_mgr_.Load(path, so_func);
  if (status != SUCCESS) {
    GELOGE(status, "[Load][EngineSo]Failed, lib path %s", path.c_str());
    REPORT_CALL_ERROR("E19999", "Load engine so failed, lib path %s", path.c_str());
    return status;
  }

  status = plugin_mgr_.InvokeAll<std::map<std::string, DNNEnginePtr> &>(so_api_func, engines_map_);
  if (status != SUCCESS) {
    GELOGE(status, "[Get][DNNEngineObjs]Failed, so_api_func %s", so_api_func.c_str());
    REPORT_CALL_ERROR("E19999", "Get DNNEngineObjs failed, so_api_func %s", so_api_func.c_str());
    return status;
  }

  GELOGI("The number of DNNEngineObjs is %zu.", engines_map_.size());

  // Engines initialize
  for (auto iter = engines_map_.begin(); iter != engines_map_.end(); ++iter) {
    if (iter->second == nullptr) {
      GELOGI("Engine: %s point to nullptr", (iter->first).c_str());
      continue;
    }

    GELOGI("DNNEngine name: %s.", (iter->first).c_str());

    status = iter->second->Initialize(options);
    if (status != SUCCESS) {
      GELOGE(status, "[Init][Engine]Failed, engine %s", (iter->first).c_str());
      REPORT_CALL_ERROR("E19999", "Initialize engine %s failed", (iter->first).c_str());
      return status;
    }


    // Check engines' attribute
    DNNEngineAttribute attrs;
    iter->second->GetAttributes(attrs);
    if (attrs.runtime_type == RuntimeType::DEVICE) {
      if ((attrs.mem_type.size()) != 1 || (attrs.mem_type[0] != GE_ENGINE_ATTR_MEM_TYPE_HBM)) {
        GELOGE(GE_ENG_MEMTYPE_ERROR, "[Check][Param]Engine %s in aicore, but the memory type is "
               "not HBM, mem_type_size %lu", (iter->first).c_str(), attrs.mem_type.size());
        REPORT_INNER_ERROR("E19999", "Engine %s in aicore, but the memory type is not HBM, "
                          "mem_type_size %lu", (iter->first).c_str(), attrs.mem_type.size());
        return GE_ENG_MEMTYPE_ERROR;
      }
    }
  }

  status = ParserJsonFile();
  if (status != SUCCESS) {
    GELOGE(status, "[Parse][JsonFile]Failed");
    return status;
  }

  status = CheckJsonFile();
  if (status != SUCCESS) {
    GELOGE(status, "[Check][JsonFile]Failed");
    return status;
  }

  init_flag_ = true;

  return SUCCESS;
}

Status DNNEngineManager::Finalize() {
  // Finalize is not allowed, initialize first is necessary
  if (!init_flag_) {
    GELOGW("DNNEngineManager has been finalized.");
    return SUCCESS;
  }

  for (auto iter = engines_map_.begin(); iter != engines_map_.end(); ++iter) {
    if (iter->second != nullptr) {
      GELOGI("DNNEngine name: %s.", (iter->first).c_str());
      Status status = iter->second->Finalize();
      if (status != SUCCESS) {
        GELOGE(status, "[Finalize][Engine]Failed, engine %s", (iter->first).c_str());
        REPORT_CALL_ERROR("E19999", "Finalize engine %s failed", (iter->first).c_str());
        return status;
      }
    }
  }
  init_flag_ = false;
  engines_map_.clear();
  return SUCCESS;
}

std::shared_ptr<ge::DNNEngine> DNNEngineManager::GetEngine(const std::string &name) const {
  auto iter = engines_map_.find(name);
  if (iter != engines_map_.end()) {
    return iter->second;
  }

  GELOGW("Failed to get engine object by engine name. %s.", name.c_str());
  return nullptr;
}

bool DNNEngineManager::IsEngineRegistered(const std::string &name) {
  auto iter = engines_map_.find(name);
  if (iter != engines_map_.end()) {
    return true;
  }
  GELOGW("Engine: %s is not Registered", name.c_str());
  return false;
}

void DNNEngineManager::InitPerformanceStaistic() {
  std::lock_guard<std::mutex> lock(mutex_);
  checksupport_cost_.clear();
}

const map<string, uint64_t> &DNNEngineManager::GetCheckSupportCost() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return checksupport_cost_;
}

std::string DNNEngineManager::GetDNNEngineName(const ge::NodePtr &node_ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GE_CLI_GE_NOT_INITIALIZED, "DNNEngineManager: node_ptr is nullptr");
                  return "");
  auto op_desc = node_ptr->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GE_CLI_GE_NOT_INITIALIZED, "DNNEngineManager: op_desc is nullptr");
                  return "");
  // Use the OpsKernelManager in GELib to get the opInfos for this opCode
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][DNNEngineName]Failed, gelib not init before");
    REPORT_INNER_ERROR("E19999", "Get DNNEngineName failed, gelib not init before");
    return "";
  }
  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  std::vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    GELOGI("DNNEngineManager: Can not get op info by op type %s", op_desc->GetType().c_str());
    return "";
  }
  GE_IF_BOOL_EXEC(ExecOnHostCpu(op_desc), return GetHostCpuEngineName(op_infos, op_desc));
  std::string ge_core_type;
  Status ret = ge::GetContext().GetOption(ge::CORE_TYPE, ge_core_type);
  GE_IF_BOOL_EXEC(ret != SUCCESS, GELOGD("get the option CORE_TYPE fail, set it to default value VECTOR_ENGINE"));
  std::string exclude_core_Type = (ge_core_type == kVectorCore) ? kAIcoreEngine : kVectorEngine;
  GELOGD("engine type will exclude: %s", exclude_core_Type.c_str());

  auto root_graph = ge::GraphUtils::FindRootGraph(node_ptr->GetOwnerComputeGraph());
  std::map<std::string, std::string> unsupported_reasons;
  for (const auto &it : op_infos) {
    if (it.engine == exclude_core_Type) {
      continue;
    }
    auto &kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      std::string unsupported_reason;
      // It will be replaced by engine' checksupport
      uint64_t start_time = GetCurrentTimestamp();
      if (kernel_info_store->second->CheckSupported(node_ptr, unsupported_reason)) {
        checksupport_cost_[kernel_name] += GetCurrentTimestamp() - start_time;
        op_desc->SetOpEngineName(it.engine);
        op_desc->SetOpKernelLibName(kernel_name);
        // set attrs for taking information when load txt to graph object
        if (it.flagAsync) {
          GELOGD("Set aicpu blocking op:%s attribute(is_blocking_op):true", op_desc->GetName().c_str());
          (void)AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
        }
        (void) AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, it.engine);
        (void) AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kernel_name);
        GELOGD("DNNEngineManager:Set OpKernelLibName %s and engine name %s to op_desc %s", kernel_name.c_str(),
               it.engine.c_str(), op_desc->GetName().c_str());
        return it.engine;
      } else {
        checksupport_cost_[kernel_name] += GetCurrentTimestamp() - start_time;
        unsupported_reasons.emplace(kernel_name, unsupported_reason);
        GELOGI("DNNEngineManager:Check support failed, kernel_name is %s, op type is %s, op name is %s",
               kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
        if (!op_desc->HasAttr("_is_ge_op")) {
          ErrorManager::GetInstance().ATCReportErrMessage("W11001", {"opname"}, {op_desc->GetName()});
        }
      }
    } else {
      GELOGW(
          "DNNEngineManager:Can not find any supported ops kernel info store by kernel_name %s,"
          "op type is %s, op name is %s",
          kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
    }
  }

  // concat unsupported reasons analyzed data selection
  string reason;
  for (const auto &it : unsupported_reasons) {
    reason += it.first + ":" + it.second + ";";
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E13002", {"optype", "opskernel", "reason"}, {op_desc->GetType(), it.first, it.second});
    GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Check][OpSupported]Op type %s of ops kernel %s "
           "is unsupported, reason : %s",
           op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str());
  }

  analyzer::DataInfo analyze_info{root_graph->GetSessionID(), root_graph->GetGraphID(),
                                  analyzer::CHECKSUPPORT, node_ptr, reason};
  // do not change original process
  (void)Analyzer::GetInstance()->DoAnalyze(analyze_info);

  ErrorManager::GetInstance().ATCReportErrMessage(
      "E13003", {"opname", "optype"}, {op_desc->GetName(), op_desc->GetType()});
  GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Get][DNNEngineName]Can't find any supported ops kernel "
         "and engine of %s, type is %s",
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return "";
}

std::string DNNEngineManager::GetHostCpuEngineName(const std::vector<OpInfo> &op_infos,
                                                   const OpDescPtr &op_desc) const {
  for (const auto &it : op_infos) {
    if ((it.engine == kHostCpuEngineName) && (it.opKernelLib == kHostCpuOpKernelLibName)) {
      op_desc->SetOpEngineName(kHostCpuEngineName);
      op_desc->SetOpKernelLibName(kHostCpuOpKernelLibName);
      GELOGI("DNNEngineManager: Set OpKernelLibName %s and OpEngineName %s to %s",
             kHostCpuOpKernelLibName, kHostCpuEngineName, op_desc->GetName().c_str());
      return kHostCpuEngineName;
    }
  }
  GELOGE(FAILED, "[Get][HostCpuEngineName]Failed, HostCpuEngine not support [%s, %s]",
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  REPORT_INNER_ERROR("E19999", "Get HostCpuEngineName failed, HostCpuEngine not support [%s, %s]",
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return "";
}

const std::map<std::string, SchedulerConf> &DNNEngineManager::GetSchedulers() const { return schedulers_; }

Status DNNEngineManager::ParserJsonFile() {
  GELOGI("Begin to parser json file");
  std::string json_file_path = "plugin/nnengine/ge_config/engine_conf.json";
  std::string path = PluginManager::GetPath();
  path.append(json_file_path);
  nlohmann::json scheduler_json_file;
  Status status = ReadJsonFile(path, &scheduler_json_file);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Read][JsonFile]Failed, file %s", path.c_str());
    REPORT_CALL_ERROR("E19999", "Read json file %s failed", path.c_str());
    return FAILED;
  }
  if (scheduler_json_file.is_null()) {
    // when engine_conf.json is not exist, just return success
    GELOGW("Json file is null");
    return SUCCESS;
  }

  try {
    nlohmann::json scheduler_utils_json = scheduler_json_file[kSchedulerUnits];
    if (scheduler_utils_json.is_null()) {
      GELOGE(FAILED, "[Check[Param]Find scheduler units failed, the message is null, file %s", path.c_str());
      REPORT_INNER_ERROR("E19999", "Find scheduler units failed, the message is null, file %s", path.c_str());
      return FAILED;
    }
    if (!scheduler_utils_json.is_array()) {
      GELOGE(FAILED, "[Check][Param]The message of kSchedulerUnits is not array and "
             "the file path is %s", path.c_str());
      REPORT_INNER_ERROR("E19999", "The message of kSchedulerUnits is not array and "
                        "the file path is %s", path.c_str());
      return FAILED;
    }
    auto size = scheduler_json_file[kSchedulerUnits].size();
    for (size_t i = 0; i < size; i++) {
      SchedulerConf scheduler_conf;
      std::map<std::string, EngineConfPtr> engine_conf_map;
      nlohmann::json engines_json_map = scheduler_utils_json[i][kCalEngines];
      if (engines_json_map.is_null()) {
        GELOGE(FAILED, "[Check][Param]The message of cal_engines is null, file %s", path.c_str());
        REPORT_INNER_ERROR("E19999", "The message of cal_engines is null, file %s", path.c_str());
        return FAILED;
      }
      std::string scheduler_id_temp = scheduler_utils_json[i][kId];
      if (!scheduler_id_temp.empty()) {
        scheduler_conf.id = scheduler_id_temp;
      } else {
        GELOGE(FAILED, "[Check][Param]Scheduler ID is null, file %s", path.c_str());
        REPORT_INNER_ERROR("E19999", "Scheduler ID is null, file %s", path.c_str());
        return FAILED;
      }
      status = ParserEngineMessage(engines_json_map, scheduler_id_temp, engine_conf_map);
      if (status != SUCCESS) {
        GELOGE(FAILED, "[Parse][EngineMessage]Failed, scheduler_id_temp %s", scheduler_id_temp.c_str());
        REPORT_CALL_ERROR("E19999", "Parse engine message failed, scheduler_id_temp %s",
                          scheduler_id_temp.c_str());
        return FAILED;
      }
      scheduler_conf.name = scheduler_utils_json[i][kName];
      scheduler_conf.ex_attrs = scheduler_utils_json[i][kExAttrs];
      scheduler_conf.cal_engines = engine_conf_map;
      auto it = schedulers_.find(scheduler_id_temp);
      if (it != schedulers_.end()) {
        GELOGE(FAILED, "[Check][Param]There are the same scheduler ts %s in the json file",
               scheduler_id_temp.c_str());
        REPORT_INNER_ERROR("E19999", "[Check][Param]There are the same scheduler ts %s "
                          "in the json file", scheduler_id_temp.c_str());
        return FAILED;
      }
      schedulers_.emplace(scheduler_id_temp, scheduler_conf);
    }
  } catch (const nlohmann::detail::type_error &e) {
    GELOGE(FAILED, "[Parse][JsonFile]Failed, file %s, reason %s", path.c_str(), e.what());
    REPORT_CALL_ERROR("E19999", "Parse json file %s failed, reason %s", path.c_str(), e.what());
    return FAILED;
  }

  GELOGI("Parser json file SUCCESS");
  return SUCCESS;
}

Status DNNEngineManager::ParserEngineMessage(const json engines_json, const std::string &scheduler_mark,
                                             std::map<std::string, EngineConfPtr> &engines) {
  GELOGI("Begin to parser engine massage");
  if (engines_json.is_null()) {
    GELOGE(FAILED, "[Check][Param]The message of cal_engines is null");
    REPORT_INNER_ERROR("E19999", "The message of cal_engines is null");
    return FAILED;
  }
  try {
    if (engines_json.is_array()) {
      for (size_t i = 0; i < engines_json.size(); i++) {
        nlohmann::json engines_elems = engines_json[i];
        EngineConfPtr engine_conf_ptr = MakeShared<EngineConf>();
        if (engine_conf_ptr == nullptr) {
          return FAILED;
        }
        std::string engine_id = engines_elems[kId];
        if (!engine_id.empty()) {
          engine_conf_ptr->id = engine_id;
        } else {
          GELOGE(FAILED, "[Check][Param]Engine ID is null");
          REPORT_INNER_ERROR("E19999", "Engine ID is null");
          return FAILED;
        }
        if (engines_elems.find(kName) != engines_elems.end()) {
          engine_conf_ptr->name = engines_elems[kName];
        } else {
          GELOGW("The engine %s name is null", engine_id.c_str());
        }
        if (engines_elems.find(kIndependent) != engines_elems.end()) {
          engine_conf_ptr->independent = engines_elems[kIndependent];
        }

        if (engines_elems.find(kAttch) != engines_elems.end()) {
          engine_conf_ptr->attach = engines_elems[kAttch];
        }

        if (engines_elems.find(kSkipAssignStream) != engines_elems.end()) {
          engine_conf_ptr->skip_assign_stream = engines_elems[kSkipAssignStream];
        }
        engine_conf_ptr->scheduler_id = scheduler_mark;
        auto it = engines.find(engine_id);
        if (it != engines.end()) {
          GELOGE(FAILED, "[Check][Param]There are the same engine %s message in the json file",
                 engine_id.c_str());
          REPORT_INNER_ERROR("E19999", "There are the same engine %s message in the json file",
                             engine_id.c_str());
          return FAILED;
        }
        engines.emplace(engine_id, engine_conf_ptr);
      }
    } else {
      GELOGE(FAILED, "[Check][Param]The message of cal_engines is not array in the json file");
      REPORT_INNER_ERROR("E19999", "The message of cal_engines is not array in the json file");
      return FAILED;
    }
  } catch (const json::exception &e) {
    GELOGE(FAILED, "[Construct][JsonContent]Failed, reason %s", e.what());
    REPORT_INNER_ERROR("E19999", "Construct json content failed, reason %s", e.what());
    return FAILED;
  }
  GELOGI("Parser engine massage success");
  return SUCCESS;
}

Status DNNEngineManager::ReadJsonFile(const std::string &file_path, JsonHandle handle) {
  GELOGD("Begin to read json file");
  if (file_path.empty()) {
    GELOGE(FAILED, "[Check][Param]Json path is empty");
    REPORT_INNER_ERROR("E19999", "Json path is empty");
    return FAILED;
  }
  nlohmann::json *json_file = reinterpret_cast<nlohmann::json *>(handle);
  if (json_file == nullptr) {
    GELOGE(FAILED, "[Check][Param]Json file is nullptr");
    REPORT_CALL_ERROR("E19999", "Json file is nullptr");
    return FAILED;
  }
  const char *file = file_path.data();
  if ((mmAccess2(file, M_F_OK)) != EN_OK) {
    if (engines_map_.size() != 0) {
      GELOGE(FAILED, "[Check][Param]The json file %s not exists, err %s",
             file_path.c_str(), strerror(errno));
      REPORT_CALL_ERROR("E19999", "Json file %s not exists, err %s",
                        file_path.c_str(), strerror(errno));
      return FAILED;
    } else {
      GELOGW("The json file %s is not needed.", file_path.c_str());
      return SUCCESS;
    }
  }

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    GELOGE(FAILED, "[Open][JsonFile]Failed, file %s", file_path.c_str());
    REPORT_CALL_ERROR("E19999", "Open json file %s failed", file_path.c_str());
    return FAILED;
  }

  try {
    ifs >> *json_file;
  } catch (const json::exception &e) {
    GELOGE(FAILED, "[Read][JsonFile]Failed, reason %s", e.what());
    REPORT_CALL_ERROR("E19999", "Read json file failed, reason %s", e.what());
    ifs.close();
    return FAILED;
  }
  ifs.close();
  GELOGD("Read json file success");
  return SUCCESS;
}

Status DNNEngineManager::CheckJsonFile() {
  GELOGD("Begin to check json file");
  for (auto &it : engines_map_) {
    std::string engine_name = it.first;
    int count = 0;
    for (auto &iter : schedulers_) {
      auto engine_map = iter.second.cal_engines;
      auto iter_engine_name = engine_map.find(engine_name);
      if (iter_engine_name != engine_map.end()) {
        count++;
      }
    }
    if (count == 0) {
      GELOGE(FAILED, "[Check][JsonFile]The engine message %s is not found in the json file",
             engine_name.c_str());
      REPORT_INNER_ERROR("E19999", "The engine message %s is not found in the json file",
                         engine_name.c_str());
      return FAILED;
    }
    if (count > 1) {
      GELOGE(FAILED, "[Check][JsonFile]The same engine message %s exists in the json file",
             engine_name.c_str());
      REPORT_INNER_ERROR("E19999", "The same engine message %s exists in the json file",
                         engine_name.c_str());
      return FAILED;
    }
  }
  GELOGD("Check json file success");
  return SUCCESS;
}
}  // namespace ge
