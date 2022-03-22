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

#include "easy_graph/graph/node.h"
#include "easy_graph/graph/graph_visitor.h"
#include <algorithm>

EG_NS_BEGIN

__DEF_EQUALS(Node) {
  return id_ == rhs.id_;
}

__DEF_COMP(Node) {
  return id_ < rhs.id_;
}

NodeId Node::GetId() const {
  return id_;
}

Node &Node::Packing(const BoxPtr &box) {
  this->box_ = box;
  return *this;
}

Node &Node::AddSubgraph(const Graph &graph) {
  subgraphs_.push_back(&graph);
  return *this;
}

void Node::Accept(GraphVisitor &visitor) const {
  std::for_each(subgraphs_.begin(), subgraphs_.end(), [&visitor](const auto &graph) { visitor.Visit(*graph); });
}

EG_NS_END
