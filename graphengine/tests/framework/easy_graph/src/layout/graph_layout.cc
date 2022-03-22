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

#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/layout_executor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

namespace {
GraphEasyExecutor default_executor;
}

void GraphLayout::Config(LayoutExecutor &executor, const LayoutOption *opts) {
  this->executor_ = &executor;
  options_ = opts;
}

Status GraphLayout::Layout(const Graph &graph, const LayoutOption *opts) {
  const LayoutOption *options = opts ? opts : this->options_;
  if (!executor_) return static_cast<LayoutExecutor &>(default_executor).Layout(graph, options);
  return executor_->Layout(graph, options);
}

EG_NS_END
