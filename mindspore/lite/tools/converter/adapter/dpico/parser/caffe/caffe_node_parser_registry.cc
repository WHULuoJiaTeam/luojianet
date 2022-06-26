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

#include "parser/caffe/caffe_node_parser_registry.h"

namespace mindspore {
namespace lite {
CaffeNodeParserRegistry::CaffeNodeParserRegistry() = default;

CaffeNodeParserRegistry::~CaffeNodeParserRegistry() {
  for (auto ite : parsers) {
    if (ite.second != nullptr) {
      delete ite.second;
      ite.second = nullptr;
    }
  }
}

CaffeNodeParserRegistry *CaffeNodeParserRegistry::GetInstance() {
  static CaffeNodeParserRegistry instance;
  return &instance;
}

CaffeNodeParser *CaffeNodeParserRegistry::GetNodeParser(const std::string &name) {
  auto it = parsers.find(name);
  if (it != parsers.end()) {
    return it->second;
  }
  return nullptr;
}
}  // namespace lite
}  // namespace mindspore
