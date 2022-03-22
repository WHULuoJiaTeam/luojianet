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

#ifndef GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_
#define GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "framework/common/debug/log.h"
#include "common/ge/plugin_manager.h"
#include "common/ge/op_tiling_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"
#include "graph/optimize/graph_optimize.h"
#include "framework/common/ge_inner_error_codes.h"
#include "external/ge/ge_api_types.h"
#include "runtime/base.h"

using std::string;
using std::map;
using std::vector;

namespace ge {
using OpsKernelInfoStorePtr = std::shared_ptr<OpsKernelInfoStore>;

class GE_FUNC_VISIBILITY OpsKernelManager {
 public:
  friend class GELib;

  // get opsKernelInfo by type
  const vector<OpInfo> &GetOpsKernelInfo(const string &op_type);

  // get all opsKernelInfo
  const map<string, vector<OpInfo>> &GetAllOpsKernelInfo() const;

  // get opsKernelInfoStore by name
  OpsKernelInfoStorePtr GetOpsKernelInfoStore(const std::string &name) const;

  // get all opsKernelInfoStore
  const map<string, OpsKernelInfoStorePtr> &GetAllOpsKernelInfoStores() const;

  // get all graph_optimizer
  const map<string, GraphOptimizerPtr> &GetAllGraphOptimizerObjs() const;

  // get all graph_optimizer by priority
  const vector<pair<string, GraphOptimizerPtr>> &GetAllGraphOptimizerObjsByPriority() const;

  // get subgraphOptimizer by engine name
  void GetGraphOptimizerByEngine(const std::string &engine_name, vector<GraphOptimizerPtr> &graph_optimizer);

  // get enableFeFlag
  bool GetEnableFeFlag() const;

  // get enableAICPUFlag
  bool GetEnableAICPUFlag() const;

  // get enablePluginFlag
  bool GetEnablePluginFlag() const;

 private:
  OpsKernelManager();
  ~OpsKernelManager();

  // opsKernelManager initialize, load all opsKernelInfoStore and graph_optimizer
  Status Initialize(const map<string, string> &options);

  // opsKernelManager finalize, unload all opsKernelInfoStore and graph_optimizer
  Status Finalize();

  Status InitOpKernelInfoStores(const map<string, string> &options);

  Status CheckPluginPtr() const;

  void GetExternalEnginePath(std::string &path, const std::map<string, string>& options);

  void InitOpsKernelInfo();

  Status InitGraphOptimzers(const map<string, string> &options);

  Status InitPluginOptions(const map<string, string> &options);

  Status ParsePluginOptions(const map<string, string> &options, const string &plugin_name, bool &enable_flag);

  Status LoadGEGraphOptimizer(map<string, GraphOptimizerPtr>& graphOptimizer);

  Status InitGraphOptimizerPriority();

  // Finalize other ops kernel resource
  Status FinalizeOpsKernel();

  PluginManager plugin_manager_;
  OpTilingManager op_tiling_manager_;
  // opsKernelInfoStore
  map<string, OpsKernelInfoStorePtr> ops_kernel_store_{};
  // graph_optimizer
  map<string, GraphOptimizerPtr> graph_optimizers_{};
  // ordered graph_optimzer
  vector<pair<string, GraphOptimizerPtr>> graph_optimizers_by_priority_{};
  // opsKernelInfo
  map<string, vector<OpInfo>> ops_kernel_info_{};

  map<string, string> initialize_{};

  vector<OpInfo> empty_op_info_{};

  bool init_flag_;

  bool enable_fe_flag_ = false;

  bool enable_aicpu_flag_ = false;
};
}  // namespace ge
#endif  // GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_
