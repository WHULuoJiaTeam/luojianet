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
#include "ir_build/attr_options/attr_options.h"
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "common/omg_util.h"
namespace ge {
  namespace {
  const std::string CFG_PRE_OPTYPE = "OpType::";
}
bool IsOriginalOpFind(OpDescPtr &op_desc, const std::string &op_name) {
  std::vector<std::string> original_op_names;
  if (!AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names)) {
    return false;
  }

  for (auto &origin_name : original_op_names) {
    if (origin_name == op_name) {
      return true;
    }
  }

  return false;
}

bool IsOpTypeEqual(const ge::NodePtr &node, const std::string &op_type) {
  if (op_type != node->GetOpDesc()->GetType()) {
    return false;
  }
  std::string origin_type;
  auto ret = GetOriginalType(node, origin_type);
  if (ret != SUCCESS) {
    GELOGW("[Get][OriginalType] from op:%s failed.", node->GetName().c_str());
    return false;
  }
  if (op_type != origin_type) {
    return false;
  }
  return true;
}

bool IsContainOpType(const std::string &cfg_line, std::string &op_type) {
  op_type = cfg_line;
  size_t pos = op_type.find(CFG_PRE_OPTYPE);
  if (pos != std::string::npos) {
    if (pos == 0) {
      op_type = cfg_line.substr(CFG_PRE_OPTYPE.length());
      return true;
    } else {
      GELOGW("[Check][Param] %s must be at zero pos of %s", CFG_PRE_OPTYPE.c_str(), cfg_line.c_str());
    }
    return false;
  }
  GELOGW("[Check][Param] %s not contain optype", cfg_line.c_str());
  return false;
}
}  // namespace ge