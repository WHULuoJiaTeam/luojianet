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
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
namespace {
const size_t kMaxOpsNum = 10;
}  // namespace

void KeepDtypeReportError(const std::vector<std::string> &invalid_list, const std::string &cfg_path) {
  std::stringstream err_msg;
  size_t list_size = invalid_list.size();
  err_msg << "config file contains " << list_size;
  if (list_size == 1) {
    err_msg << " operator not in the graph, ";
  } else {
    err_msg << " operators not in the graph, ";
  }
  std::string cft_type;
  for (size_t i = 0; i < list_size; i++) {
    if (i == kMaxOpsNum) {
      err_msg << "..";
      break;
    }
    bool istype = IsContainOpType(invalid_list[i], cft_type);
    if (!istype) {
      err_msg << "op name:";
    } else {
      err_msg << "op type:";
    }
    err_msg << cft_type;
    if (i != (list_size - 1)) {
      err_msg << " ";
    }
  }

  REPORT_INPUT_ERROR(
      "E10003", std::vector<std::string>({"parameter", "value", "reason"}),
      std::vector<std::string>({"keep_dtype", cfg_path, err_msg.str()}));
  GELOGE(FAILED, "%s", err_msg.str().c_str());
}

graphStatus KeepDtypeFunc(ComputeGraphPtr &graph, const std::string &cfg_path) {
  GE_CHECK_NOTNULL(graph);
  if (cfg_path.empty()) {
    return GRAPH_SUCCESS;
  }
  std::string real_path = RealPath(cfg_path.c_str());
  if (real_path.empty()) {
    GELOGE(GRAPH_PARAM_INVALID, "[Get][Path]Can not get real path for %s.", cfg_path.c_str());
    REPORT_INPUT_ERROR("E10410", std::vector<std::string>({"cfgpath"}), std::vector<std::string>({cfg_path}));
    return GRAPH_PARAM_INVALID;
  }
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed.", cfg_path.c_str());
    REPORT_INNER_ERROR("E19999", "open file:%s failed.", cfg_path.c_str());
    return GRAPH_FAILED;
  }

  std::string op_name, op_type;
  std::vector<std::string> invalid_list;
  while (std::getline(ifs, op_name)) {
    if (op_name.empty()) {
      continue;
    }
    op_name = StringUtils::Trim(op_name);
    bool is_find = false;
    bool is_type = IsContainOpType(op_name, op_type);
    for (auto &node_ptr : graph->GetAllNodes()) {
      auto op_desc = node_ptr->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (is_type) {
        if (IsOpTypeEqual(node_ptr, op_type)) {
          is_find = true;
          (void)AttrUtils::SetInt(op_desc, ATTR_NAME_KEEP_DTYPE, 1);
        }
      } else {
        if (op_desc->GetName() == op_name || IsOriginalOpFind(op_desc, op_name)) {
          is_find = true;
          (void)AttrUtils::SetInt(op_desc, ATTR_NAME_KEEP_DTYPE, 1);
        }
      }
    }
    if (!is_find) {
      invalid_list.push_back(op_name);
    }
  }
  ifs.close();

  if (!invalid_list.empty()) {
    KeepDtypeReportError(invalid_list, cfg_path);
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge
