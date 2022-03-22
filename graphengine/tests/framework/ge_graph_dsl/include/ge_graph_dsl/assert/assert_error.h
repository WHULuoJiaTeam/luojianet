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
#ifndef D52AA06185E34BBFB714FFBCDAB0D53A
#define D52AA06185E34BBFB714FFBCDAB0D53A

#include "ge_graph_dsl/ge.h"
#include <exception>
#include <string>

GE_NS_BEGIN

struct AssertError : std::exception {
  AssertError(const char *file, int line, const std::string &info);

 private:
  const char *what() const noexcept override;

 private:
  std::string info;
};

GE_NS_END

#endif