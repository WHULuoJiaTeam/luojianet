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

#ifndef HB6783151_C24E_4DA3_B969_46C2298FF43F
#define HB6783151_C24E_4DA3_B969_46C2298FF43F

#include <string>
#include "easy_graph/graph/graph_visitor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_layout_context.h"

EG_NS_BEGIN

struct GraphEasyOption;

struct GraphEasyVisitor : GraphVisitor {
  GraphEasyVisitor(const GraphEasyOption &);

  std::string GetLayout() const;

 private:
  Status Visit(const Graph &) override;
  Status Visit(const Node &) override;
  Status Visit(const Edge &) override;

 private:
  std::string layout_;
  GraphEasyLayoutContext ctxt_;
};

EG_NS_END

#endif
