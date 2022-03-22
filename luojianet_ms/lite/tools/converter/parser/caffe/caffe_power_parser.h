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

#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_POWER_PARSER_H_
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_POWER_PARSER_H_

#include <vector>
#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"

namespace luojianet_ms {
namespace lite {
class CaffePowerParser : public CaffeNodeParser {
 public:
  CaffePowerParser() : CaffeNodeParser("power") {}
  ~CaffePowerParser() override = default;

  ops::PrimitiveC *Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};
}  // namespace lite
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_POWER_PARSER_H_
