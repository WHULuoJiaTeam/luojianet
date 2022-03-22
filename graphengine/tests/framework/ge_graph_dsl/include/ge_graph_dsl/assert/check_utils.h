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
#ifndef INC_31309AA0A4E44C009C22AD9351BF3410
#define INC_31309AA0A4E44C009C22AD9351BF3410

#include "ge_graph_dsl/ge.h"
#include "graph/compute_graph.h"

GE_NS_BEGIN

using GraphCheckFun = std::function<void(const ::GE_NS::ComputeGraphPtr &)>;
struct CheckUtils {
  static bool CheckGraph(const std::string &phase_id, const GraphCheckFun &fun);
  static void init();
};

GE_NS_END

#endif