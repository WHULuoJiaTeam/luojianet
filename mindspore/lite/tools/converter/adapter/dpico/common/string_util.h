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

#ifndef DPICO_COMMON_STRING_UTIL_H_
#define DPICO_COMMON_STRING_UTIL_H_

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;

namespace mindspore {
namespace dpico {
int EraseBlankSpace(std::string *input_string);
int EraseHeadTailSpace(std::string *input_string);
std::vector<std::string> SplitString(const std::string &raw_str, char delimiter);
std::string RemoveSpecifiedChar(const std::string &origin_str, char specified_ch);
std::string ReplaceSpecifiedChar(const std::string &origin_str, char origin_ch, char target_ch);
bool IsValidUnsignedNum(const std::string &num_str);
bool IsValidDoubleNum(const std::string &num_str);
}  // namespace dpico
}  // namespace mindspore
#endif  // DPICO_COMMON_STRING_UTIL_H_
