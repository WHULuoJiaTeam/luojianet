/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_LOGICAL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_LOGICAL_PARSER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser.h"

namespace mindspore {
namespace lite {
class TFLogicalAndParser : public TFNodeParser {
 public:
  TFLogicalAndParser() = default;
  ~TFLogicalAndParser() override = default;

  PrimitiveCPtr Parse(const tensorflow::NodeDef &tf_op,
                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                      std::vector<std::string> *inputs, int *output_size) override;
};

class TFLogicalOrParser : public TFNodeParser {
 public:
  TFLogicalOrParser() = default;
  ~TFLogicalOrParser() override = default;

  PrimitiveCPtr Parse(const tensorflow::NodeDef &tf_op,
                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                      std::vector<std::string> *inputs, int *output_size) override;
};

class TFLogicalNotParser : public TFNodeParser {
 public:
  TFLogicalNotParser() = default;
  ~TFLogicalNotParser() override = default;

  PrimitiveCPtr Parse(const tensorflow::NodeDef &tf_op,
                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                      std::vector<std::string> *inputs, int *output_size) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_LOGICAL_PARSER_H_
