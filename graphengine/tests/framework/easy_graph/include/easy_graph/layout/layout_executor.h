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

#ifndef HE4984335_C586_4533_A056_27F9F996DF50
#define HE4984335_C586_4533_A056_27F9F996DF50

#include "easy_graph/infra/status.h"
#include "easy_graph/infra/keywords.h"

EG_NS_BEGIN

struct Graph;
struct LayoutOption;

INTERFACE(LayoutExecutor) {
  ABSTRACT(Status Layout(const Graph &, const LayoutOption *));
};

EG_NS_END

#endif
