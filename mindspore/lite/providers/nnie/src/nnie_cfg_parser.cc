/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/nnie_cfg_parser.h"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/nnie_manager.h"
#include "src/nnie_print.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace nnie {
namespace {
constexpr auto kTimeStep = "TimeStep";
constexpr auto kMazRoiNum = "MaxROINum";
constexpr auto kCoreIds = "CoreIds";
constexpr auto DELIM = ",";
constexpr int MAX_CORE_ID = 7;
}  // namespace
bool IsValidUnsignedNum(const std::string &num_str) {
  return !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit);
}

void PrintInvalidChar(const std::string &key, const std::string &dat) {
  auto message = key + " configuration contains invalid characters: \'" + dat + "\'";
  LOGE(message.c_str());
}

int Flags::Init(const kernel::Kernel &kernel) {
  auto nnie_arg = kernel.GetConfig("nnie");
  if (nnie_arg.find(kTimeStep) != nnie_arg.end()) {
    if (IsValidUnsignedNum(nnie_arg.at(kTimeStep)) == true) {
      this->time_step_ = stoi(nnie_arg.at(kTimeStep));
    } else {
      PrintInvalidChar(kTimeStep, nnie_arg.at(kTimeStep));
      return RET_ERROR;
    }
  }

  if (nnie_arg.find(kMazRoiNum) != nnie_arg.end()) {
    if (IsValidUnsignedNum(nnie_arg.at(kMazRoiNum)) == true) {
      this->max_roi_num_ = stoi(nnie_arg.at(kMazRoiNum));
    } else {
      PrintInvalidChar(kMazRoiNum, nnie_arg.at(kMazRoiNum));
      return RET_ERROR;
    }
  }

  if (nnie_arg.find(kCoreIds) != nnie_arg.end()) {
    auto ids = nnie_arg.at(kCoreIds);
    if (!ids.empty() && std::all_of(ids.begin(), ids.end(), [](char val) { return ::isdigit(val) || val == ','; })) {
      size_t index = 0;
      std::vector<int> core_ids;
      while (!ids.empty()) {
        core_ids.push_back(std::stoi(ids, &index));
        if (index < ids.length()) {
          ids = ids.substr(index + 1);
        } else {
          break;
        }
      }
      this->core_ids_ = core_ids;
    } else {
      PrintInvalidChar(kCoreIds, ids);
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace nnie
}  // namespace mindspore
