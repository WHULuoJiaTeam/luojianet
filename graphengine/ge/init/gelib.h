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

#ifndef GE_INIT_GELIB_H_
#define GE_INIT_GELIB_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "engine_manager/dnnengine_manager.h"
#include "opskernel_manager/ops_kernel_manager.h"
#include "graph/tuning_utils.h"
#include "graph/operator_factory.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/anchor_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"

using std::string;
using std::map;
using std::vector;

namespace ge {
class GE_FUNC_VISIBILITY GELib {
 public:
  GELib() = default;
  ~GELib() = default;

  // get GELib singleton instance
  static std::shared_ptr<GELib> GetInstance();

  // GE Environment Initialize, return Status: SUCCESS,FAILED
  static Status Initialize(const map<string, string> &options);

  static string GetPath();

  // GE Environment Finalize, return Status: SUCCESS,FAILED
  Status Finalize();

  // get DNNEngineManager object
  DNNEngineManager &DNNEngineManagerObj() { return engineManager_; }

  // get OpsKernelManager object
  OpsKernelManager &OpsKernelManagerObj() { return opsManager_; }

  // get Initial flag
  bool InitFlag() const { return init_flag_; }

  // get TrainMode flag
  bool IsTrainMode() { return is_train_mode_; }

  void InitProfiling(Options &options);
  void ShutDownProfiling();

  Status InitSystemWithoutOptions();
  Status InitSystemWithOptions(Options &options);
  Status SystemShutdownWithOptions(const Options &options);

 private:
  GELib(const GELib &);
  const GELib &operator=(const GELib &);
  Status InnerInitialize(const map<string, string> &options);
  Status SystemInitialize(const map<string, string> &options);
  Status SetRTSocVersion(const map<string, string> &options, map<string, string> &new_options);
  Status SetAiCoreNum(map<string, string> &options);
  void SetDefaultPrecisionMode(map<string, string> &new_options);
  void RollbackInit();
  void InitOptions(const map<string, string> &options);
  void SetDumpModelOptions(const map<string, string> &options);
  void SetOpDebugOptions(const map<string, string> &options);

  DNNEngineManager engineManager_;
  OpsKernelManager opsManager_;
  std::mutex status_mutex_;
  bool init_flag_ = false;
  Options options_;
  bool is_train_mode_ = false;
  bool is_system_inited = false;
  bool is_shutdown = false;
  bool is_use_hcom = false;
};
}  // namespace ge

#endif  // GE_INIT_GELIB_H_
