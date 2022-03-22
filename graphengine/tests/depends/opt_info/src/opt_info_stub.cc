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
#include "opt_info.h"
#include <string>
#include <map>
#include <vector>
#include <algorithm>

namespace gelc {
namespace {
const std::vector<std::string> kSocVersions = {"Ascend910"};
}

void SetAllOptInfo(std::map<std::string, std::string> &opt_infos) {
  opt_infos.emplace("opt_module.fe", "all");
  opt_infos.emplace("opt_module.pass", "all");
  opt_infos.emplace("opt_module.op_tune", "all");
  opt_infos.emplace("opt_module.rl_tune", "all");
  opt_infos.emplace("opt_module.aoe", "all");
}

Status GetOptInfo(WorkMode mode, const std::string &soc_ver,
                  std::map<std::string, std::string> &opt_infos) {
  if (std::find(kSocVersions.begin(), kSocVersions.end(), soc_ver)== kSocVersions.end()) {
    SetAllOptInfo(opt_infos);
    return SUCCESS;
  }
  opt_infos.emplace("opt_module.fe", "all");
  opt_infos.emplace("opt_module.pass", "all");
  opt_infos.emplace("opt_module.op_tune", "all");
  return SUCCESS;
}
}  // namespace gelc
