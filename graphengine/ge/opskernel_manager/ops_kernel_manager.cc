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

#include "opskernel_manager/ops_kernel_manager.h"

#include "init/gelib.h"
#include "proto/optimizer_priority.pb.h"

namespace {
const char *const kInitialize = "Initialize";
const char *const kGetOpsKernelInfoStores = "GetOpsKernelInfoStores";
const char *const kGetGraphOptimizerObjs = "GetGraphOptimizerObjs";
const char *const kFinalize = "Finalize";

std::mutex ops_kernel_info_mutex;
}  // namespace

namespace ge {
OpsKernelManager::OpsKernelManager()
    : plugin_manager_(), op_tiling_manager_(), init_flag_(false), enable_fe_flag_(false), enable_aicpu_flag_(false) {}

OpsKernelManager::~OpsKernelManager() {
  graph_optimizers_.clear();
  ops_kernel_store_.clear();
  ops_kernel_info_.clear();
}

Status OpsKernelManager::Initialize(const map<string, string> &options_const) {
  if (init_flag_) {
    GELOGW("OpsKernelManager has been initialized.");
    return SUCCESS;
  }
  std::map<string, string> options(options_const);
  Status ret = InitPluginOptions(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][PluginOptions] parse pluginFlag from ge options failed.");
    REPORT_CALL_ERROR("E19999", "InitPluginOptions failed.");
    return ret;
  }

  vector<string> func_check_list = {kInitialize, kGetOpsKernelInfoStores, kGetGraphOptimizerObjs, kFinalize};
  string extern_engine_path;

  auto iter = options.find(OPTION_EXEC_IS_USEHCOM);
  if (iter == options.end()) {
    GELOGI("OPTION_EXEC_IS_USEHCOM is not set, default is single P");
    options.emplace("ge.exec.isUseHcom", to_string(0));
  }

  iter = options.find(OPTION_EXEC_IS_USEHVD);
  if (iter == options.end()) {
    GELOGI("OPTION_EXEC_IS_USEHVD is not set, default is single P");
    options.emplace("ge.exec.isUseHvd", to_string(0));
  }

  GetExternalEnginePath(extern_engine_path, options);
  GELOGI("OPTION_EXEC_EXTERN_PLUGIN_PATH=%s.", extern_engine_path.c_str());

  op_tiling_manager_.LoadSo();

  ret = plugin_manager_.LoadSo(extern_engine_path, func_check_list);
  if (ret == SUCCESS) {
    initialize_ = options;
    Status rst0 = plugin_manager_.InvokeAll<map<string, string> &, Status>(kInitialize, initialize_);
    if (rst0 == FAILED) {
      GELOGE(GE_OPS_GET_NO_VALID_SO, "[Invoke][OpsKernelInfo]PluginManager InvokeAll failed.");
      REPORT_INNER_ERROR("E19999", "PluginManager InvokeAll failed.");
      return GE_OPS_GET_NO_VALID_SO;
    }
    Status rst1 =
        plugin_manager_.InvokeAll<map<string, OpsKernelInfoStorePtr> &>(kGetOpsKernelInfoStores, ops_kernel_store_);
    if (rst1 != SUCCESS) {
      GELOGW("Initialize OpsKernelInfo failed.");
    }
    Status rst2 =
        plugin_manager_.InvokeAll<map<string, GraphOptimizerPtr> &>(kGetGraphOptimizerObjs, graph_optimizers_);
    if (rst2 != SUCCESS) {
      GELOGW("Initialize GraphOptimizerObjs failed.");
    }

    ret = CheckPluginPtr();
    if (ret != SUCCESS) {
      return ret;
    }
    ret = InitOpKernelInfoStores(options);
    if (ret != SUCCESS) {
      return ret;
    }
    InitOpsKernelInfo();
    ret = InitGraphOptimzers(options);
    if (ret != SUCCESS) {
      return ret;
    }
    ret = InitGraphOptimizerPriority();
    if ((ret != SUCCESS)) {
      GELOGE(ret, "[Init][GraphOptimizerPriority] failed.");
      REPORT_CALL_ERROR("E19999", "InitGraphOptimizerPriority failed.");
      return ret;
    }
    init_flag_ = true;
    return SUCCESS;
  } else {
    GELOGE(ret, "[Check][SoFile] not find any valid so file.");
    REPORT_INNER_ERROR("E19999", "OpsKernelManager::Initialize failed for not find any valid so file.");
    return ret;
  }
}

void OpsKernelManager::GetExternalEnginePath(std::string &extern_engine_path,
    const std::map<string, string>& options) {
  GELOGI("Enter get external engine so path schedule");
  const char *path_env = std::getenv("ASCEND_ENGINE_PATH");
  if (path_env != nullptr) {
    extern_engine_path = path_env;
    GELOGI("OpsKernelManager get external engine so path from env.");
    return;
  }
  std::string path_base = PluginManager::GetPath();
  std::string so_path = "plugin/opskernel/";
  std::string path = path_base + so_path;
  extern_engine_path = (path + "libfe.so" + ":") + (path + "libge_local_engine.so" + ":") +
                       (path + "librts_engine.so" + ":") + (path + "libaicpu_ascend_engine.so" + ":") +
                       (path + "libhost_cpu_engine.so" + ":") + (path + "libaicpu_tf_engine.so" + ":");
  auto iter = options.find(OPTION_EXEC_HCCL_FLAG);
  if (iter == options.end() || iter->second != "0") {
    extern_engine_path += (path_base + "libhcom_graph_adaptor.so");
  }
}

Status OpsKernelManager::InitPluginOptions(const map<string, string> &options) {
  Status ret;

  // parse fe
  ret = ParsePluginOptions(options, GE_FE_FLAG, enable_fe_flag_);
  if (ret != SUCCESS) {
    return ret;
  }

  // parse aiCpu
  ret = ParsePluginOptions(options, GE_AICPU_FLAG, enable_aicpu_flag_);
  if (ret != SUCCESS) {
    return ret;
  }

  return SUCCESS;
}

Status OpsKernelManager::ParsePluginOptions(const map<string, string> &options, const string &plugin_name,
                                            bool &enable_flag) {
  GELOGI("Parse the Plugin Options, plugin_name:%s.", plugin_name.c_str());
  auto iter = options.find(plugin_name);
  if (iter != options.end()) {
    try {
      int32_t flag = std::stoi(iter->second.c_str());
      if (flag == 0) {
        enable_flag = false;
      } else if (flag == 1) {
        enable_flag = true;
      } else {
        GELOGE(GE_GRAPH_OPTIONS_INVALID,
               "[Parse][PluginOptions]option_key:%s, its value %s is invalid, it must be 0 or 1.",
               plugin_name.c_str(), iter->second.c_str());
        REPORT_INNER_ERROR("E19999", "ParsePluginOptions failed, option_key:%s, "
                           "its value %s is invalid, it must be 0 or 1.", plugin_name.c_str(), iter->second.c_str());
        return GE_GRAPH_OPTIONS_INVALID;
      }
    } catch (std::invalid_argument &) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][PluginOptions] failed, option_key:ge.feFlag,"
             "its value %s is invalid_argument, it must be 0 or 1.",
             iter->second.c_str());
      REPORT_INNER_ERROR("E19999", "ParsePluginOptions failed, option_key:ge.feFlag,"
                         "its value %s is invalid_argument, it must be 0 or 1.",
                         iter->second.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    } catch (std::out_of_range &) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID,
             "[Parse][PluginOptions]failed, option_key:ge.feFlag, its value %s is out of range, it must be 0 or 1.",
             iter->second.c_str());
      REPORT_INNER_ERROR("E19999", "ParsePluginOptions failed, option_key:ge.feFlag,"
                         "its value %s is out of range, it must be 0 or 1.",
                         iter->second.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    } catch (...) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID,
             "[Parse][PluginOptions]option_key:%s, its value %s is invalid, it must be 0 or 1.",
             plugin_name.c_str(), iter->second.c_str());
      REPORT_INNER_ERROR("E19999", "ParsePluginOptions failed, option_key:%s, "
                         "its value %s is invalid, it must be 0 or 1.", plugin_name.c_str(), iter->second.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  } else {
    GELOGI("Not find option_key %s, set to default value false.", plugin_name.c_str());
    enable_flag = false;
  }

  return SUCCESS;
}

Status OpsKernelManager::CheckPluginPtr() const {
  for (auto iter = ops_kernel_store_.begin(); iter != ops_kernel_store_.end(); ++iter) {
    if (iter->second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][PluginPtr] OpsKernelInfoStorePtr key=%s is null", iter->first.c_str());
      REPORT_INNER_ERROR("E19999", "CheckPluginPtr OpsKernelInfoStorePtr key=%s is null", iter->first.c_str());
      return FAILED;
    }
  }
  for (auto iter1 = graph_optimizers_.begin(); iter1 != graph_optimizers_.end(); ++iter1) {
    if (iter1->second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][PluginPtr] GraphOptimizerPtr key=%s is null", iter1->first.c_str());
      REPORT_INNER_ERROR("E19999", "GraphOptimizerPtr key=%s is null", iter1->first.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpsKernelManager::InitOpKernelInfoStores(const map<string, string> &options) {
  GELOGI("The number of OpKernelInfoStoreObjs are %lu.", ops_kernel_store_.size());
  for (const auto &it : ops_kernel_store_) {
    GELOGI("OpKernelInfoStore name: %s.", (it.first).c_str());
    Status ret = it.second->Initialize(options);
    if (ret != SUCCESS) {
      GELOGE(GE_OPS_KERNEL_STORE_INIT_FAILED,
             "[Init][OpKernelLib]OpKernelInfoStore: %s initialize failed.", (it.first).c_str());
      REPORT_CALL_ERROR("E19999", "OpKernelInfoStore: %s initialize failed.", (it.first).c_str());
      return GE_OPS_KERNEL_STORE_INIT_FAILED;
    }
  }

  return SUCCESS;
}

void OpsKernelManager::InitOpsKernelInfo() {
  ops_kernel_info_.clear();
  for (const auto &it : ops_kernel_store_) {
    map<string, OpInfo> op_infos{};
    it.second->GetAllOpsKernelInfo(op_infos);
    for (const auto &op_info_it : op_infos) {
      auto op_info_copy = op_info_it.second;
      // flush ops kernel
      op_info_copy.opKernelLib = it.first;
      ops_kernel_info_[op_info_it.first].emplace_back(op_info_copy);
      GELOGD("OpKernelInfoStore name: %s, found op type is %s, engine name is %s, opkernel name is %s",
             (it.first).c_str(), op_info_it.first.c_str(), op_info_it.second.engine.c_str(),
             op_info_it.second.opKernelLib.c_str());
    }
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib]malloc instance_ptr failed.");
    REPORT_INNER_ERROR("E19999", "InitOpsKernelInfo failed for new GELib.");
    return;
  }
  // sort opinfo of ops_kernel_info_
  for (auto &it : ops_kernel_info_) {
    if (it.second.empty()) {
      continue;
    }
    auto comp_func = [&instance_ptr](const OpInfo &op_a, const OpInfo &op_b) -> bool {
      const string &a = op_a.engine;
      const string &b = op_b.engine;
      // check if a or b is registered
      if (!(instance_ptr->DNNEngineManagerObj().IsEngineRegistered(a))) {
        return false;
      }
      if (!(instance_ptr->DNNEngineManagerObj().IsEngineRegistered(b))) {
        return true;
      }
      // compare compute cost of a and b, IsEngineRegistered make sure engine is not nullptr
      auto engine_a = instance_ptr->DNNEngineManagerObj().GetEngine(a);
      auto engine_b = instance_ptr->DNNEngineManagerObj().GetEngine(b);
      DNNEngineAttribute attr_a, attr_b;
      engine_a->GetAttributes(attr_a);
      engine_b->GetAttributes(attr_b);
      return attr_a.compute_cost < attr_b.compute_cost;
    };
    // Sort the OpInfos based on the compute cost of the engine
    std::sort(it.second.begin(), it.second.end(), comp_func);
  }
  GELOGI("Init opsKernelInfo finished, size is %zu", ops_kernel_info_.size());
}

Status OpsKernelManager::InitGraphOptimzers(const map<string, string> &options) {
  GELOGI("Init graph optimizers options count %zu", options.size());
  for (const auto &option : options) {
    GELOGI("Init graph optimizers option %s: %s", option.first.c_str(), option.second.c_str());
  }
  GELOGI("The number of GraphOptimzerObjs are %zu.", graph_optimizers_.size());
  for (const auto &it : graph_optimizers_) {
    GELOGI("GraphOptimzer name: %s.", (it.first).c_str());
    GraphOptimizerAttribute attrs;
    GE_CHK_STATUS_RET(it.second->GetAttributes(attrs))
    std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr == nullptr) {
      GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib]malloc instance_ptr failed.");
      REPORT_INNER_ERROR("E19999", "InitGraphOptimzers failed for new GELib.");
      return GE_CLI_GE_NOT_INITIALIZED;
    }
    if (!instance_ptr->DNNEngineManagerObj().IsEngineRegistered(attrs.engineName)) {
      GELOGW("Engine: %s is not registered.", attrs.engineName.c_str());
      continue;
    }
    Status ret = it.second->Initialize(options);
    if (ret != SUCCESS) {
      GELOGE(GE_OPS_GRAPH_OPTIMIZER_INIT_FAILED, 
          "[Init][GraphOptimzer]GraphOptimzer: %s initialize failed.", (it.first).c_str());
      REPORT_CALL_ERROR("E19999", "InitGraphOptimzers failed. %s initialize failed.", (it.first).c_str());
      return GE_OPS_GRAPH_OPTIMIZER_INIT_FAILED;
    }
  }
  return SUCCESS;
}

Status OpsKernelManager::Finalize() {
  if (!init_flag_) {
    GELOGW("Finalize is not allowed, initialize first is necessary.");
    return SUCCESS;
  }
  GELOGI("free ops kernel resource.");
  for (auto iter = ops_kernel_store_.begin(); iter != ops_kernel_store_.end(); ++iter) {
    GELOGI("OpsKernelStore finalize, name: %s.", (iter->first).c_str());
    Status status = iter->second->Finalize();
    if (SUCCESS != status) {
      GELOGE(status, "[Check][Status]OpsKernelStore finalize failed, name: %s.", (iter->first).c_str());
      REPORT_CALL_ERROR("E19999", "OpsKernelStore finalize failed, name: %s.", (iter->first).c_str());
      return status;
    }
  }
  for (auto iter = graph_optimizers_.begin(); iter != graph_optimizers_.end(); ++iter) {
    GELOGI("GraphOptimzers finalize, name: %s.", (iter->first).c_str());
    Status status = iter->second->Finalize();
    if (status != SUCCESS) {
      GELOGE(status, "[Check][Status]GraphOptimzers finalize failed, name: %s.", (iter->first).c_str());
      REPORT_CALL_ERROR("E19999", "GraphOptimzers finalize failed, name: %s.", (iter->first).c_str());
      return status;
    }
  }

  Status ret = FinalizeOpsKernel();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Free][Ops Kernel Resource] failed.");
    REPORT_CALL_ERROR("E19999", "FinalizeOpsKernel failed, Free Ops kernel resource failed.");
    return ret;
  }

  init_flag_ = false;
  return SUCCESS;
}

const vector<OpInfo> &OpsKernelManager::GetOpsKernelInfo(const string &op_type) {
  std::lock_guard<std::mutex> lock(ops_kernel_info_mutex);

  auto find = ops_kernel_info_.find(op_type);
  if (find != ops_kernel_info_.end()) {
    return find->second;
  } else {
    InitOpsKernelInfo();
    find = ops_kernel_info_.find(op_type);
    if (find != ops_kernel_info_.end()) {
      return find->second;
    }
    GELOGW("Failed to get opsKernelInfo object by type: %s.", op_type.c_str());
    return empty_op_info_;
  }
}

const map<string, vector<OpInfo>> &OpsKernelManager::GetAllOpsKernelInfo() const {
  std::lock_guard<std::mutex> lock(ops_kernel_info_mutex);
  return ops_kernel_info_;
}

OpsKernelInfoStorePtr OpsKernelManager::GetOpsKernelInfoStore(const std::string &name) const {
  auto find = ops_kernel_store_.find(name);
  if (find != ops_kernel_store_.end()) {
    return find->second;
  }

  GELOGW("Failed to get opsKernelInfoStore object by name. OpKernelLibName is %s", name.c_str());
  return nullptr;
}

const map<string, OpsKernelInfoStorePtr> &OpsKernelManager::GetAllOpsKernelInfoStores() const {
  return ops_kernel_store_;
}

const map<string, GraphOptimizerPtr> &OpsKernelManager::GetAllGraphOptimizerObjs() const { return graph_optimizers_; }

const vector<pair<string, GraphOptimizerPtr>> &OpsKernelManager::GetAllGraphOptimizerObjsByPriority() const {
  return graph_optimizers_by_priority_;
}

void OpsKernelManager::GetGraphOptimizerByEngine(const std::string &engine_name,
                                                 vector<GraphOptimizerPtr> &graph_optimizer) {
  for (const auto &it : graph_optimizers_) {
    GraphOptimizerAttribute attrs;
    if (it.second->GetAttributes(attrs) != SUCCESS) {
      GELOGW("Get GraphOptimzer name: %s attributes failed.", (it.first).c_str());
      continue;
    }
    if (attrs.engineName == engine_name) {
      GELOGD("GetGraphOptimizerByEngine GraphOptimzer name: %s, engineName: %s", (it.first).c_str(),
             attrs.engineName.c_str());
      graph_optimizer.push_back(it.second);
    }
  }

  if (graph_optimizer.empty()) {
    GELOGI("GetGraphOptimizerByEngine EngineName %s has no graph_optimizer.", engine_name.c_str());
  }
}

bool OpsKernelManager::GetEnableFeFlag() const { return enable_fe_flag_; }

bool OpsKernelManager::GetEnableAICPUFlag() const { return enable_aicpu_flag_; }

bool OpsKernelManager::GetEnablePluginFlag() const { return (enable_fe_flag_ || enable_aicpu_flag_); }

Status OpsKernelManager::InitGraphOptimizerPriority() {
  string priority_conf_path = "plugin/opskernel/optimizer_priority.pbtxt";
  string path = PluginManager::GetPath();
  path.append(priority_conf_path);

  optimizers::Priority optimizerPriority;
  bool ret = ReadProtoFromText(path.c_str(), &optimizerPriority);
  if (!ret) {
    GELOGW("Read priority file failed. Follow loading sequence.");
    return SUCCESS;
  }
  auto priorities = optimizerPriority.optimizer();
  if (priorities.empty()) {
    GELOGI("No priority file config. Follow loading sequence.");
    return SUCCESS;
  }
  // sort optimizer map by priority
  std::stringstream priority_seq;
  for (const auto optimizer_name : priorities) {
    auto name_to_optimizer_pair = graph_optimizers_.find(optimizer_name);
    if (name_to_optimizer_pair != graph_optimizers_.end()) {
      graph_optimizers_by_priority_.emplace_back(*name_to_optimizer_pair);
      priority_seq << optimizer_name.c_str() << ' ';
    } else {
      GELOGW("Unknown optimizer %s show up in priority config file. Please check.", optimizer_name.c_str());
    }
  }
  GELOGI("Graph Optimizers priority initialized. The sequence will follow : %s.", priority_seq.str().c_str());
  return SUCCESS;
}

Status OpsKernelManager::FinalizeOpsKernel() {
  GELOGI("ge invoke ops kernal finalize.");
  Status ret = plugin_manager_.InvokeAll<Status>(kFinalize);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Finalize][Check][Status] invoke Fe finalize failed.");
    REPORT_INNER_ERROR("E19999", "PluginManager InvokeAll failed.");
    return ret;
  }

  return SUCCESS;
}
}  // namespace ge
