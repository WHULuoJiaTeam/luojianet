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
#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_TF_CONV_BASE_PARSER_H_
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_TF_CONV_BASE_PARSER_H_
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser.h"

namespace luojianet_ms {
namespace lite {
class TFConvBaseParser : public TFNodeParser {
 public:
  TFConvBaseParser() = default;
  ~TFConvBaseParser() override = default;

  static STATUS ParseStrides(const tensorflow::NodeDef &node_def, const luojianet_ms::Format &format,
                             std::vector<int64_t> *stridstatices);
  static STATUS ParseDilations(const tensorflow::NodeDef &node_def, const luojianet_ms::Format &format,
                               std::vector<int64_t> *dilations);
  static STATUS ParseKernels(const tensorflow::NodeDef &node_def, const luojianet_ms::Format &format,
                             std::vector<int64_t> *kernel);
  static STATUS ParseExplicitPaddings(const tensorflow::NodeDef &node_def, const luojianet_ms::Format &format,
                                      std::vector<int64_t> *explicit_paddings);
  static luojianet_ms::PadMode ParsePadMode(const tensorflow::NodeDef &node_def, bool *is_ori_pad_mode);
};
}  // namespace lite
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_TF_TF_CONV_BASE_PARSER_H_
