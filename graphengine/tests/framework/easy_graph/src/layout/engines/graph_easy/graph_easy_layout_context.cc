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

#include <algorithm>
#include "easy_graph/layout/engines/graph_easy/graph_easy_layout_context.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

GraphEasyLayoutContext::GraphEasyLayoutContext(const GraphEasyOption &options) : options_(options) {}

const Graph *GraphEasyLayoutContext::GetCurrentGraph() const {
  if (graphs_.empty())
    return nullptr;
  return graphs_.back();
}

void GraphEasyLayoutContext::EnterGraph(const Graph &graph) {
  graphs_.push_back(&graph);
}

void GraphEasyLayoutContext::ExitGraph() {
  graphs_.pop_back();
}

void GraphEasyLayoutContext::LinkBegin() {
  is_linking_ = true;
}

void GraphEasyLayoutContext::LinkEnd() {
  is_linking_ = false;
}

bool GraphEasyLayoutContext::InLinking() const {
  return is_linking_;
}

std::string GraphEasyLayoutContext::GetGroupPath() const {
  if (graphs_.empty())
    return "";
  std::string result("");
  std::for_each(graphs_.begin(), graphs_.end(),
                [&result](const auto &graph) { result += (std::string("/") + graph->GetName()); });
  return (result + "/");
}

const GraphEasyOption &GraphEasyLayoutContext::GetOptions() const {
  return options_;
}

EG_NS_END
