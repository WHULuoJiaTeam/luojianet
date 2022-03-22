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

#ifndef GE_GRAPH_MANAGER_UTIL_VARIABLE_ACCELERATE_CTRL_H_
#define GE_GRAPH_MANAGER_UTIL_VARIABLE_ACCELERATE_CTRL_H_

#include <map>
#include <set>
#include <string>
#include <mutex>

#include "graph/compute_graph.h"
#include "graph/node.h"

namespace ge {
class VarAccelerateCtrl {
 public:
  void AddGraph(uint32_t graph_id, const ComputeGraphPtr &compute_graph);

  void RemoveGraph(uint32_t graph_id);

  void SetVarChanged(const std::string &var_name);

  bool IsGraphNeedRebuild(uint32_t graph_id) const;

  void SetGraphBuildEnd(uint32_t graph_id);

  bool IsVarPermitToChangeFormats(const std::string &var_name);

 private:
  ///
  /// the variable and graph relationships will construct when `AddGraph`
  ///
  std::map<uint32_t, std::set<std::string>> graph_ids_to_var_names_;

  ///
  /// The graph id of the graph to be rebuilt. When the format of a variable is
  /// changed, the graph which contains this variable is needs to be rebuilt.
  ///
  std::set<uint32_t> graph_ids_need_rebuild_;

  ///
  /// Number of variable names and their format changes.
  /// In order to prevent the variable format from being repeatedly changed
  /// between different formats, we simply limited the variable format to
  /// only one time of changing
  ///
  std::map<std::string, int> var_names_to_change_times_;
  static const int kMaxVarChangeTimes_ = 1;

  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_VARIABLE_ACCELERATE_CTRL_H_
