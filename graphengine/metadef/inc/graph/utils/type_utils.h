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

#ifndef INC_GRAPH_UTILS_TYPE_UTILS_H_
#define INC_GRAPH_UTILS_TYPE_UTILS_H_

#include <map>
#include <set>
#include <unordered_set>
#include <string>
#include "graph/def_types.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "register/register_types.h"
#include "external/register/register_fmk_types.h"

namespace ge {
class TypeUtils {
 public:
  static bool IsDataTypeValid(const DataType dt);
  static bool IsFormatValid(const Format format);
  static bool IsDataTypeValid(std::string dt); // for user json input
  static bool IsFormatValid(std::string format); // for user json input
  static bool IsInternalFormat(const Format format);

  static std::string ImplyTypeToSerialString(const domi::ImplyType imply_type);
  static std::string DataTypeToSerialString(const DataType data_type);
  static DataType SerialStringToDataType(const std::string &str);
  static std::string FormatToSerialString(const Format format);
  static Format SerialStringToFormat(const std::string &str);
  static Format DataFormatToFormat(const std::string &str);
  static graphStatus SplitFormatFromStr(const std::string &str, std::string &primary_format_str, int32_t &sub_format);
  static Format DomiFormatToFormat(const domi::domiTensorFormat_t domi_format);
  static std::string FmkTypeToSerialString(const domi::FrameworkType fmk_type);

  static bool GetDataTypeLength(const ge::DataType data_type, uint32_t &length);
  static bool CheckUint64MulOverflow(const uint64_t a, const uint32_t b);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TYPE_UTILS_H_
