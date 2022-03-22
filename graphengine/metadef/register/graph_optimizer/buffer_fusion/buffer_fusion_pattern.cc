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

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include <string>
#include <vector>
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

using std::map;
using std::string;
using std::vector;

namespace fe {
inline bool IsAddOverflow(int64_t a, int64_t b) {
  return ((b > 0) && (a > ((int64_t)INT64_MAX - b))) || ((b < 0) && (a < ((int64_t)INT64_MIN - b)));
}

BufferFusionPattern::BufferFusionPattern(string name, int64_t max_count)
    : name_(name), op_max_count_(max_count), error_count_(0) {}

BufferFusionPattern::~BufferFusionPattern() {
  for (auto op : ops_) {
    if (op == nullptr) {
      continue;
    }
    delete (op);
  }
}

/*
 * @brief:  add op desc info
 * @param [in] desc_name: node desc name
 * @param [in] types: node desc type
 * @param [in] repeate_min: the min count for fusion match,
 *                         patter match failed if real count lower than the
 * value
 * @param [in] repeate_max: the max count for fusion match,
 *                         the op will be ignored if current match count equal
 * with the value
 * @return BufferFusionPattern: pattern object
 */
BufferFusionPattern &BufferFusionPattern::AddOpDesc(const std::string &desc_name, const std::vector<std::string> &types,
                                                    int64_t repeate_min, int64_t repeate_max, int64_t group_id,
                                                    ShapeTypeRule shape_type_rule, bool not_pattern) {
  if (desc_name.empty()) {
    GELOGW("[AddOpDesc][Check] Desc_name cannot be empty.");
    error_count_++;
    return *this;
  }

  if (repeate_min > repeate_max) {
    GELOGW("[AddOpDesc][Check] Check desc %s failed as repeat_min > repeat_max, repeat_min=%ld, repeat_max=%ld",
           desc_name.c_str(), repeate_min, repeate_max);
    error_count_++;
    return *this;
  }

  if (GetOpDesc(desc_name) != nullptr) {
    GELOGW("[AddOpDesc][Check] Desc_name repeated. (desc_name:%s)", desc_name.c_str());
    error_count_++;
    return *this;
  }

  BufferFusionOpDesc *op = new (std::nothrow) BufferFusionOpDesc();
  if (op == nullptr) {
    GELOGW("[AddOpDesc][Check] New an object failed.");
    error_count_++;
    return *this;
  }

  op->desc_name = desc_name;
  op->types = types;
  op->repeate_min = repeate_min;
  op->repeate_max = repeate_max;
  op->repeate_curr = 0;
  op->group_id = group_id;
  op->shape_type_rule = shape_type_rule;
  op->match_status = false;
  op->out_branch_type = TBE_OUTPUT_BRANCH_DEFAULT;
  op->ignore_input_num = false;
  op->ignore_output_num = false;
  op->not_pattern = not_pattern;
  if (repeate_max > repeate_min) {
    for (int64_t i = repeate_min; i < repeate_max; i++) {
      op->multi_output_skip_status.insert(std::pair<int64_t, SkipStatus>(i, SkipStatus::DISABLED));
    }
  }
  ops_.push_back(op);
  op_map_[desc_name] = op;

  op->outputs.clear();
  return *this;
}

/*
 * @brief:  set output desc info
 * @param [in] desc_name: node desc name
 * @param [in] output_ids: output desc
 * @param [in] relation:   output desc relation (1: serial, 2:parallel)
 * @return BufferFusionPattern: pattern object
 */
BufferFusionPattern &BufferFusionPattern::SetOutputs(const string &desc_name, const std::vector<string> &output_ids,
                                                     int64_t relation, bool ignore_input_num, bool ignore_output_num) {
  if (desc_name.empty()) {
    GELOGW("[SetOutputs][Check] Desc_name cannot be empty.");
    error_count_++;
    return *this;
  }

  BufferFusionOpDesc *op_desc = GetOpDesc(desc_name);
  if (op_desc == nullptr) {
    GELOGW("[SetOutputs][Check] Desc_name %s not exist", desc_name.c_str());
    error_count_++;
    return *this;
  }

  op_desc->ignore_input_num = ignore_input_num;
  op_desc->ignore_output_num = ignore_output_num;
  if (op_desc->out_branch_type == TBE_OUTPUT_BRANCH_DEFAULT) {
    op_desc->out_branch_type = relation;
  }

  UpdateSkipStatus(op_desc);

  // support one multi output for one op_type
  for (const string &output_id : output_ids) {
    BufferFusionOpDesc *output_op_desc = GetOpDesc(output_id);
    if (output_op_desc == nullptr) {
      GELOGW("[SetOutputs][Check] Desc_name not exist. (desc_name:%s)", desc_name.c_str());
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetOutputs][Check] errorCount_++ overflow. (desc_name:%s)", desc_name.c_str());
        return *this;
      }
      error_count_++;
      return *this;
    }
    if (op_desc == output_op_desc) {
      continue;
    }

    op_desc->outputs.push_back(output_op_desc);
    output_op_desc->inputs.push_back(op_desc);

    if (op_desc->out_branch_type != relation) {
      GELOGW("[SetOutputs][Check] Set outputs relation failed, curr is [%ld], new is [%ld].", op_desc->out_branch_type,
             relation);
      return *this;
    }
  }
  return *this;
}

/*
 * @brief:  get output desc info
 * @param [in]  op_desc: current desc
 * @param [out] outputs: candidate output desc set
 * @return bool: get output desc ok or not
 */
bool BufferFusionPattern::GetOutputs(BufferFusionOpDesc *op_desc, std::vector<BufferFusionOpDesc *> &outputs,
                                     bool ignore_repeat) {
  if (op_desc == nullptr) {
    GELOGW("[GetOutputs][Check] op_desc is null.");
    return false;
  }
  string desc_n = op_desc->desc_name;

  // add curr desc can be reused while repeat_curr < repeate_max
  if (!ignore_repeat && op_desc->repeate_curr < op_desc->repeate_max) {
    outputs.push_back(op_desc);
  }

  // check candidate desc
  for (auto desc : op_desc->outputs) {
    if (desc == nullptr) {
      continue;
    }
    // add out desc
    outputs.push_back(desc);

    // add sub out_descs while repeate_min == 0
    if (desc->repeate_min == 0) {
      std::vector<BufferFusionOpDesc *> sub_output;
      if (GetOutputs(desc, sub_output, true)) {
        for (const auto &sub_desc : sub_output) {
          outputs.push_back(sub_desc);
        }
      }
    }
  }

  return true;
}

/*
 * @brief: set fusion pattern head
 * @param [in] head_ids: node list
 * @return bool: set head desc ok or not
 */
BufferFusionPattern &BufferFusionPattern::SetHead(const std::vector<string> &head_ids) {
  if (head_ids.empty()) {
    GELOGW("[SetHead][Check] Input head_ids is empty.");
    error_count_++;
    return *this;
  }
  for (const string &head_id : head_ids) {
    BufferFusionOpDesc *head_op_desc = GetOpDesc(head_id);
    if (head_op_desc == nullptr) {
      GELOGW("[SetHead][Check] descName not exist. (desc_name:%s)", head_id.c_str());
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetHead][Check] errorCount_++ overflow. (desc_name:%s)", head_id.c_str());
        return *this;
      }
      error_count_++;
      return *this;
    }
    // Head desc repeat number can not exceed 1
    // if must be exceed 1, it can be realized by several descs
    if (head_op_desc->repeate_max > 1) {
      GELOGW("[SetHead][Check] Head desc named %s repeats more than once, cur_repeat_max=%ld", head_id.c_str(),
             head_op_desc->repeate_max);
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetHead][Check] errorCount_++ overflow. (desc_name:%s)", head_id.c_str());
        return *this;
      }
      error_count_++;
      return *this;
    }
    head_.push_back(head_op_desc);
  }

  // check head desc repeat min total value, it can not excceed 1
  int64_t desc_total_min = 0;
  for (const auto &desc : head_) {
    if (IsAddOverflow(desc_total_min, desc->repeate_min) != SUCCESS) {
      GELOGW("[SetHead][Check] desc_total_min[%ld] + repeate_min[%ld] overflow", desc_total_min, desc->repeate_min);
      return *this;
    }
    desc_total_min += desc->repeate_min;
  }

  if (desc_total_min > 1) {
    GELOGW("[SetHead][Check] Head desc repeat min total value can not be larger than 1, current is [%ld]",
           desc_total_min);
    error_count_++;
    return *this;
  }
  return *this;
}

void BufferFusionPattern::UpdateSkipStatus(BufferFusionOpDesc *op_desc) {
  if (op_desc->out_branch_type == TBE_OUTPUT_BRANCH_MULTI) {
    for (auto &input_desc : op_desc->inputs) {
      if (input_desc->types.size() != op_desc->types.size()) {
        continue;
      }
      bool is_same_type = true;
      for (size_t i = 0; i < input_desc->types.size(); i++) {
        if (input_desc->types[i] != op_desc->types[i]) {
          is_same_type = false;
          break;
        }
      }
      if (is_same_type && input_desc->ignore_output_num == true) {
        for (int64_t i = input_desc->repeate_min; i < input_desc->repeate_max; i++) {
          input_desc->multi_output_skip_status[i] = SkipStatus::AVAILABLE;
        }
      }
    }
  }
}

/*
 * @brief: get description ptr by name
 * @param [in] desc_name: fusion pattern desc name
 * @return BufferFusionOpDesc*: description ptr
 */
BufferFusionOpDesc *BufferFusionPattern::GetOpDesc(const string &desc_name) {
  auto it = op_map_.find(desc_name);
  if (it != op_map_.end()) return it->second;

  return nullptr;
}

std::vector<BufferFusionOpDesc *> BufferFusionPattern::GetHead() { return head_; }

std::string BufferFusionPattern::GetName() { return name_; }
int64_t BufferFusionPattern::GetOpMaxCount() { return op_max_count_; }
int64_t BufferFusionPattern::GetErrorCnt() { return error_count_; }

std::vector<BufferFusionOpDesc *> BufferFusionPattern::GetOpDescs() { return ops_; }
}  // namespace fe
