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

#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_UTIL_H
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_UTIL_H

#include <string>
#include <string_view>
#include "proto/node_def.pb.h"
#include "ir/dtype/type_id.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "luojianet_ms/core/utils/check_convert_utils.h"

namespace luojianet_ms {
namespace lite {
class TensorFlowUtils {
 public:
  static TypeId GetTFDataType(const tensorflow::DataType &tf_data_type);
  static bool FindAttrValue(const tensorflow::NodeDef &node_def, const std::string &attr_name,
                            tensorflow::AttrValue *attr_value);
  static TypeId ParseAttrDataType(const tensorflow::NodeDef &node_def, const std::string &attr_name);
  static bool DecodeInt64(std::string_view *str_view, uint64_t *value);
  static std::string GetFlattenNodeName(const std::string &input_name);
  static std::string GetNodeName(const std::string &input_name);
  static luojianet_ms::Format ParseNodeFormat(const tensorflow::NodeDef &node_def);
};
}  // namespace lite
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_UTIL_H
