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

#ifndef GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_
#define GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "nlohmann/json.hpp"

#include "common/ge/plugin_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/engine/dnnengine.h"
#include "graph/op_desc.h"
#include "graph/node.h"

using JsonHandle = void *;
namespace ge {
using nlohmann::json;

// Engine configuration
struct EngineConf {
  string id;                       // engine ID
  string name;                     // engine name
  bool independent{false};         // independent stream
  bool attach{false};              // attach stream
  bool skip_assign_stream{false};  // not assign stream
  string scheduler_id;             // scheduler ID
};
using EngineConfPtr = std::shared_ptr<EngineConf>;

// Configuration information of schedule unit
struct SchedulerConf {
  string id;                               // scheduler ID
  string name;                             // scheduler name
  string ex_attrs;                         // extra information
  map<string, EngineConfPtr> cal_engines;  // engine information
};

using DNNEnginePtr = std::shared_ptr<DNNEngine>;

class DNNEngineManager {
 public:
  friend class GELib;
  std::shared_ptr<ge::DNNEngine> GetEngine(const std::string &name) const;
  bool IsEngineRegistered(const std::string &name);
  // If can't find appropriate engine name, return "", report error
  string GetDNNEngineName(const ge::NodePtr &node_ptr);
  const map<string, SchedulerConf> &GetSchedulers() const;
  const map<string, uint64_t> &GetCheckSupportCost() const;
  void InitPerformanceStaistic();

 private:
  DNNEngineManager();
  ~DNNEngineManager();
  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  Status ReadJsonFile(const std::string &file_path, JsonHandle handle);
  Status ParserJsonFile();
  Status ParserEngineMessage(const json engines_json, const string &scheduler_mark,
                             map<string, EngineConfPtr> &engines);
  Status CheckJsonFile();
  std::string GetHostCpuEngineName(const std::vector<OpInfo> &op_infos, const OpDescPtr &op_desc) const;
  PluginManager plugin_mgr_;
  std::map<std::string, DNNEnginePtr> engines_map_;
  std::map<std::string, ge::DNNEngineAttribute> engines_attrs_map_;
  std::map<string, SchedulerConf> schedulers_;
  std::map<string, uint64_t> checksupport_cost_;
  bool init_flag_;
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_
