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

#include "external/register/scope/scope_fusion_pass_register.h"

#include <stdexcept>

#include "framework/common/debug/ge_log.h"
#include "framework/common/string_util.h"

namespace ge {
std::string ScopeUtil::StringReplaceAll(std::string str, const std::string &old_value, const std::string &new_value) {
  return ge::StringUtils::ReplaceAll(str, old_value, new_value);
}

AscendString ScopeUtil::StringReplaceAll(const char *str, const char *old_value, const char *new_value) {
  std::string tmp_str;
  if (str != nullptr) {
    tmp_str = str;
  }
  std::string tmp_old_value;
  if (old_value != nullptr) {
    tmp_old_value = old_value;
  }
  std::string tmp_new_value;
  if (new_value != nullptr) {
    tmp_new_value = new_value;
  }
  std::string ret = ge::StringUtils::ReplaceAll(tmp_str, tmp_old_value, tmp_new_value);
  return AscendString(ret.c_str());
}

void ScopeUtil::FreeScopePatterns(ScopeFusionPatterns &patterns) {
  for (auto &batch_pattern : patterns) {
    FreeOneBatchPattern(batch_pattern);
  }
  patterns.clear();
}

void ScopeUtil::FreeOneBatchPattern(std::vector<ScopePattern *> &one_batch_pattern) {
  for (auto &one_pattern : one_batch_pattern) {
    if (one_pattern != nullptr) {
      delete one_pattern;
      one_pattern = nullptr;
    }
  }
  one_batch_pattern.clear();
}
}  // namespace ge
