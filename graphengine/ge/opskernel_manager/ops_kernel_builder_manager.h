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

#ifndef GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_
#define GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_

#include "common/ge/plugin_manager.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "external/ge/ge_api_error_codes.h"

namespace ge {
using OpsKernelBuilderPtr = std::shared_ptr<OpsKernelBuilder>;
class GE_FUNC_VISIBILITY OpsKernelBuilderManager {
 public:
  ~OpsKernelBuilderManager();

  static OpsKernelBuilderManager& Instance();

  // opsKernelManager initialize, load all opsKernelInfoStore and graph_optimizer
  Status Initialize(const std::map<std::string, std::string> &options, bool is_train = true);

  // opsKernelManager finalize, unload all opsKernelInfoStore and graph_optimizer
  Status Finalize();

  // get opsKernelIBuilder by name
  OpsKernelBuilderPtr GetOpsKernelBuilder(const std::string &name) const;

  // get all opsKernelBuilders
  const std::map<string, OpsKernelBuilderPtr> &GetAllOpsKernelBuilders() const;

  Status CalcOpRunningParam(Node &node) const;

  Status GenerateTask(const Node &node, RunContext &context,
                      std::vector<domi::TaskDef> &tasks) const;

 private:
  OpsKernelBuilderManager() = default;
  static Status GetLibPaths(const std::map<std::string, std::string> &options, std::string &lib_paths);

  std::unique_ptr<PluginManager> plugin_manager_;
  std::map<std::string, OpsKernelBuilderPtr> ops_kernel_builders_{};
};
}  // namespace ge
#endif  // GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_
