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

#include "ge_graph_dsl/assert/assert_error.h"

GE_NS_BEGIN

AssertError::AssertError(const char *file, int line, const std::string &info) {
  this->info = std::string(file) + ":" + std::to_string(line) + "\n" + info;
}

const char *AssertError::what() const noexcept { return info.c_str(); }

GE_NS_END
