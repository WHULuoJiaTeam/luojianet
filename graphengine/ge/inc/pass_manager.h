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

#ifndef GE_INC_PASS_MANAGER_H_
#define GE_INC_PASS_MANAGER_H_

#include <vector>

#include "inc/graph_pass.h"

using std::vector;

namespace ge {
///
/// @ingroup domi_omg
/// @brief pass manager
/// @author
///
class PassManager {
 public:
  ///
  /// get graph passes
  /// @author
  ///
  const vector<std::pair<std::string, GraphPass *>> &GraphPasses() const;

  ///
  /// Add graph pass
  /// @param [in] pass  Pass to be added, it will be destroyed when pass manager destroys.
  /// @author
  ///
  Status AddPass(const string &pass_name, GraphPass *pass);

  ///
  /// Optimize graph with added pass
  /// @param [inout] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimize failed
  /// @author
  ///
  Status Run(const ge::ComputeGraphPtr &graph);

  ///
  /// Optimize graph with specified pass
  /// @param [inout] graph graph to be optimized
  /// @param [in] passes passes to be used
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  /// @author
  ///
  static Status Run(const ge::ComputeGraphPtr &graph, vector<std::pair<std::string, GraphPass *>> &passes);

  ~PassManager();

 private:
  vector<std::pair<std::string, GraphPass *>> names_to_graph_passes_;
};
}  // namespace ge
#endif  // GE_INC_PASS_MANAGER_H_
