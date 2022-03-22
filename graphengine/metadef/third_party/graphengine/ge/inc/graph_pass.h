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

#ifndef GE_INC_GRAPH_PASS_H_
#define GE_INC_GRAPH_PASS_H_

#include <string>
#include <vector>

#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "inc/pass.h"

namespace ge {
///
/// @ingroup domi_omg
/// @brief graph pass
/// @author
///
class GraphPass : public Pass<ge::ComputeGraph> {
 public:
  ///
  /// run graph pass
  /// @param [in] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  /// @author
  ///
  virtual Status Run(ge::ComputeGraphPtr graph) = 0;
  virtual Status ClearStatus() { return SUCCESS; };
  static void RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node) {
    GE_CHECK_NOTNULL_JUST_RETURN(node);
    std::vector<std::string> original_names;
    for (ge::NodePtr &node_tmp : original_nodes) {
      std::vector<std::string> names_tmp;
      ge::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
      GE_CHECK_NOTNULL_JUST_RETURN(opdesc_tmp);
      Status ret = ge::AttrUtils::GetListStr(opdesc_tmp, "_datadump_original_op_names", names_tmp);
      if (ret != domi::SUCCESS) {
        GELOGW("get the original_op_names fail.");
      }
      if (names_tmp.size() != 0) {
        original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
      } else {
        original_names.push_back(opdesc_tmp->GetName());
      }
    }

    if (original_names.size() == 0) {
      std::string tmp;
      original_names.push_back(tmp);
    }
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), "_datadump_original_op_names", original_names),
                     return, "Set original_op_names fail.");
  }

  static bool IsConstNode(const ge::NodePtr &node) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, GELOGE(FAILED, "Node GetOpDesc is nullptr"); return false);
    if (node->GetOpDesc()->GetType() == CONSTANTOP) {
      return true;
    } else if (node->GetOpDesc()->GetType() == FRAMEWORKOP) {
      string type;
      GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type),
                       return false, "Get original_type for op %s fail!", node->GetName().c_str());
      GE_IF_BOOL_EXEC(type == CONSTANT, GELOGI("Is const op"); return true);
      return false;
    } else {
      return false;
    }
  }
};
}  // namespace ge

#endif  // GE_INC_GRAPH_PASS_H_
