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
#include "opskernel_manager/ops_kernel_builder_manager.h"
#include "register/ops_kernel_builder_registry.h"

namespace ge {
namespace {
#ifdef ONLY_COMPILE_OPEN_SRC
const std::vector<std::string> kBasicBuilderLibs = {
    "libge_local_opskernel_builder.so",
    "libhost_cpu_opskernel_builder.so",
    "librts_kernel_builder.so",
    "libaicpu_ascend_builder.so",
    "libaicpu_tf_builder.so"
};
#else
const std::vector<std::string> kBasicBuilderLibs = {
    "libge_local_opskernel_builder.so",
    "libhost_cpu_opskernel_builder.so",
    "librts_engine.so",
    "libaicpu_ascend_engine.so",
    "libaicpu_tf_engine.so"
};
#endif

const std::vector<std::string> kHcclBuilderLibs = {
    "libhcom_opskernel_builder.so",
    "libhvd_opskernel_builder.so",
    "libhcom_gradtune_opskernel_builder.so"
};
}  // namespace
OpsKernelBuilderManager::~OpsKernelBuilderManager() {
  // it's OK to call Finalize multiply times
  (void) Finalize();
}

OpsKernelBuilderManager &OpsKernelBuilderManager::Instance() {
  static OpsKernelBuilderManager instance;
  return instance;
}

Status OpsKernelBuilderManager::Initialize(const map<std::string, std::string> &options, bool is_train) {
  if (is_train) {
    std::string lib_paths;
    GE_CHK_STATUS_RET_NOLOG(GetLibPaths(options, lib_paths));
    plugin_manager_.reset(new (std::nothrow)PluginManager());
    GE_CHECK_NOTNULL(plugin_manager_);
    GE_CHK_STATUS_RET(plugin_manager_->LoadSo(lib_paths),
        "[Load][Libs]Failed, lib_paths=%s.", lib_paths.c_str());
  }

  auto &kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  GELOGI("[Show][OpsKernelBuilderNum]Number of OpBuild = %zu", kernel_builders.size());

  for (const auto &it : kernel_builders) {
    const std::string &kernel_lib_name = it.first;
    GELOGI("Initialize ops kernel util for %s", kernel_lib_name.c_str());
    GE_CHECK_NOTNULL(it.second);
    GE_CHK_STATUS_RET(it.second->Initialize(options),
        "[Invoke][Initialize]failed, kernel lib name = %s", kernel_lib_name.c_str());

    ops_kernel_builders_.emplace(kernel_lib_name, it.second);
  }

  return SUCCESS;
}

Status OpsKernelBuilderManager::Finalize() {
  for (const auto &it : ops_kernel_builders_) {
    const std::string &kernel_lib_name = it.first;
    GELOGI("Finalize ops kernel util for %s", kernel_lib_name.c_str());
    auto ret = it.second->Finalize();
    if (ret != SUCCESS) {
      GELOGW("Failed to invoke Finalize, kernel lib name = %s",
             kernel_lib_name.c_str());
    }
  }

  ops_kernel_builders_.clear();
  plugin_manager_.reset();
  return SUCCESS;
}

const map<string, OpsKernelBuilderPtr> &OpsKernelBuilderManager::GetAllOpsKernelBuilders() const {
  return ops_kernel_builders_;
}

OpsKernelBuilderPtr OpsKernelBuilderManager::GetOpsKernelBuilder(const string &name) const {
  auto it = ops_kernel_builders_.find(name);
  if (it != ops_kernel_builders_.end()) {
    return it->second;
  }

  GELOGW("Failed to get opsKernelInfoStore object by name. OpKernelLibName is %s", name.c_str());
  return nullptr;
}

Status OpsKernelBuilderManager::GetLibPaths(const std::map<std::string,
                                            std::string> &options, std::string &lib_paths) {
  GELOGD("Start to execute GetLibPaths");
  std::string path_base = PluginManager::GetPath();
  std::string so_path = "plugin/opskernel/";
  std::string path = path_base + so_path;
  std::string all_lib_paths;
  for (const auto &lib_name : kBasicBuilderLibs) {
    all_lib_paths += (path + lib_name + ":");
  }

  auto iter = options.find(OPTION_EXEC_HCCL_FLAG);
  if (iter == options.end() || iter->second != "0") {
    for (const auto &lib_name : kHcclBuilderLibs) {
      all_lib_paths += (path + lib_name + ":");
    }
  }

  lib_paths = std::move(all_lib_paths);
  GELOGI("Get lib paths by default. paths = %s", lib_paths.c_str());
  return SUCCESS;
}

Status OpsKernelBuilderManager::CalcOpRunningParam(Node &node) const {
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const std::string &lib_name = op_desc->GetOpKernelLibName();
  auto it = ops_kernel_builders_.find(lib_name);
  if (it == ops_kernel_builders_.end()) {
    GELOGE(INTERNAL_ERROR,"[Find][LibName] fail for libName = %s, node = %s.",
           lib_name.c_str(), op_desc->GetName().c_str());
    REPORT_INNER_ERROR("E19999",
                       "find LibName for CalcOpRunningParam failed, libName = %s, node = %s not exist.",
                       lib_name.c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("To invoke CalcOpRunningParam, node = %s, lib name = %s", op_desc->GetName().c_str(), lib_name.c_str());
  GE_CHK_STATUS_RET(it->second->CalcOpRunningParam(node),
      "[Invoke][CalcOpRunningParam]failed, libName = %s, node = %s", lib_name.c_str(), op_desc->GetName().c_str());
  GELOGD("Done invoking CalcOpRunningParam successfully");
  return SUCCESS;
}

Status OpsKernelBuilderManager::GenerateTask(const Node &node,
                                             RunContext &context,
                                             std::vector<domi::TaskDef> &tasks) const {
  auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const std::string &lib_name = op_desc->GetOpKernelLibName();
  auto it = ops_kernel_builders_.find(lib_name);
  if (it == ops_kernel_builders_.end()) {
    GELOGE(INTERNAL_ERROR, "[Find][LibName]fail for libName = %s, node:%s", lib_name.c_str(),
           op_desc->GetName().c_str());
    REPORT_INNER_ERROR("E19999", "find LibName for GenerateTask failed, libName = %s, node = %s not exist",
                       lib_name.c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("To invoke GenerateTask, node = %s, lib name = %s", op_desc->GetName().c_str(), lib_name.c_str());
  GE_CHK_STATUS_RET(it->second->GenerateTask(node, context, tasks),
      "[Invoke][GenerateTask]failed, libName = %s, node = %s", lib_name.c_str(), op_desc->GetName().c_str());
  GELOGD("Done invoking GenerateTask successfully");
  return SUCCESS;
}
}  // namespace ge
