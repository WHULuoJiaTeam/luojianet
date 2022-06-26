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

#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_REGISTRY_H_
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_REGISTRY_H_

#include <string>
#include <unordered_map>
#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include "proto/caffe.pb.h"

namespace luojianet_ms::lite {
class CaffeNodeParserRegistry {
 public:
  static CaffeNodeParserRegistry *GetInstance();

  CaffeNodeParser *GetNodeParser(const std::string &name);

  std::unordered_map<std::string, CaffeNodeParser *> parsers;

 private:
  CaffeNodeParserRegistry();

  virtual ~CaffeNodeParserRegistry();
};

class CaffeNodeRegistrar {
 public:
  CaffeNodeRegistrar(const std::string &name, CaffeNodeParser *parser) {
    CaffeNodeParserRegistry::GetInstance()->parsers[name] = parser;
  }
  ~CaffeNodeRegistrar() = default;
};
}  // namespace luojianet_ms::lite

#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_REGISTRY_H_
