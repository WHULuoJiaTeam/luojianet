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
#ifndef INC_5960A8F437324904BEE0690271258762
#define INC_5960A8F437324904BEE0690271258762

#include "ge_graph_dsl/ge.h"
#include "easy_graph/infra/keywords.h"
#include "graph/compute_graph.h"

GE_NS_BEGIN

INTERFACE(GeGraphChecker) {
  ABSTRACT(const std::string &PhaseId() const);
  ABSTRACT(void Check(const ge::ComputeGraphPtr &graph) const);
};

GE_NS_END

#endif