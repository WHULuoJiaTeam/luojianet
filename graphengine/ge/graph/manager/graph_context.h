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

#ifndef GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_
#define GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
class GraphContext;

using SessionId = uint64_t;

using GradOpList = std::vector<std::pair<GraphId, std::string>>;

using VariableRecord = std::tuple<std::string, GradOpList, uint8_t>;

using OutputOpNameIndex = std::pair<std::string, uint8_t>;

struct key_hash : public std::unary_function<const ge::OutputOpNameIndex, std::size_t> {
  std::size_t operator()(const ge::OutputOpNameIndex &outputOpNameIndex) const {
    return (static_cast<uint8_t>(outputOpNameIndex.first[0])) ^ outputOpNameIndex.second;
  }
};

struct key_equal : public std::binary_function<const ge::OutputOpNameIndex, const ge::OutputOpNameIndex, bool> {
  bool operator()(const ge::OutputOpNameIndex &varR1, const ge::OutputOpNameIndex &varR2) const {
    return (varR1.first == varR2.first && varR1.second == varR2.second);
  }
};

using VarNodeTensorTable = std::vector<std::pair<VariableRecord, GeTensor>>;

using SessionVarTableMap = std::map<ge::SessionId, VarNodeTensorTable>;

using GraphContextPtr = std::shared_ptr<GraphContext>;

struct OutputDescInfo {
  std::string op_name;
  uint8_t index;
  struct InputOutputDescInfo info;
};

///
/// @ingroup graph
/// @brief Global graph context sharing, provide variable sharing facility for
///        multiple graphs in the same session.
/// @author
///
class GraphContext {
 public:
  GraphContext() = default;

  ~GraphContext() = default;

  Status Initialize(const std::map<std::string, std::string> &options = {}) const;
  // Disable copy constructor and assignment operator
  GraphContext(const GraphContext &) = delete;

  GraphContext &operator=(const GraphContext &) = delete;

  Status Finalize() const;

  Status GetVariableTensor(const std::string &var_data_name, GeTensor &returned_tensor);

  const ComputeGraphPtr &GetComputeGraph() const { return compute_graph_; }

  Status SetComputeGraph(const GraphNodePtr &graph_node);

 private:
  explicit GraphContext(const GraphNodePtr &graph_node);

  ComputeGraphPtr compute_graph_ = nullptr;

  GraphId current_graph_id_ = 0;

  // Get the unique VarNode-Tensor table
  static VarNodeTensorTable &GetVarNodeTensorTable() {
    static VarNodeTensorTable _this;
    return _this;
  }
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_
