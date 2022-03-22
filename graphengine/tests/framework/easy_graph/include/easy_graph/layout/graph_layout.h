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

#ifndef H550E4ACB_BEC7_4E71_8C6F_CD7FA53662A9
#define H550E4ACB_BEC7_4E71_8C6F_CD7FA53662A9

#include "easy_graph/infra/status.h"
#include "easy_graph/infra/singleton.h"

EG_NS_BEGIN

struct LayoutExecutor;
struct LayoutOption;
struct Graph;

SINGLETON(GraphLayout) {
  void Config(LayoutExecutor &, const LayoutOption * = nullptr);
  Status Layout(const Graph &, const LayoutOption * = nullptr);

 private:
  LayoutExecutor *executor_{nullptr};
  const LayoutOption *options_{nullptr};
};

EG_NS_END

#endif
