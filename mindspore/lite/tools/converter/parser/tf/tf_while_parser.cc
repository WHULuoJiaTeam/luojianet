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
#include "tools/converter/parser/tf/tf_while_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/ops/while.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFWhileParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_shared<While>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = tf_op.input_size();
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim;
}

TFNodeRegistrar g_tfStatelessWhileParser("StatelessWhile", new TFWhileParser());
TFNodeRegistrar g_tfWhileParser("While", new TFWhileParser());
}  // namespace lite
}  // namespace mindspore
