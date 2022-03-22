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

#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_visitor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "layout/engines/graph_easy/utils/shell_executor.h"
#include "easy_graph/layout/layout_option.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

namespace {
const GraphEasyOption *GraphEasyOptionCast(const LayoutOption *opts) {
  if (!opts)
    return &(GraphEasyOption::GetDefault());
  auto options = dynamic_cast<const GraphEasyOption *>(opts);
  if (options)
    return options;
  return &(GraphEasyOption::GetDefault());
}
}  // namespace

Status GraphEasyExecutor::Layout(const Graph &graph, const LayoutOption *opts) {
  auto options = GraphEasyOptionCast(opts);
  GraphEasyVisitor visitor(*options);
  graph.Accept(visitor);

  std::string script =
      std::string("echo \"") + visitor.GetLayout() + "\" | graph-easy " + options->GetLayoutCmdArgs(graph.GetName());
  return ShellExecutor::execute(script);
}

EG_NS_END
