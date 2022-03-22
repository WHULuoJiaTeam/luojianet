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
#ifndef DBF6CE7CD4AC4A83BA4ED4B372FC66E4
#define DBF6CE7CD4AC4A83BA4ED4B372FC66E4

#include "ge_running_env/fake_ns.h"
#include "graph/operator_factory.h"

FAKE_NS_BEGIN

struct FakeOpRepo {
  static void Reset();
  static void Regist(const std::string &operator_type, const OpCreator);
  static void Regist(const std::string &operator_type, const InferShapeFunc);
};

FAKE_NS_END
#endif