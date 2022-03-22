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

#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/graph_visitor.h"
#include "easy_graph/layout/graph_layout.h"
#include <algorithm>

EG_NS_BEGIN

Graph::Graph(const std::string &name) : name_(name) {}

std::string Graph::GetName() const {
  return name_;
}

Node *Graph::AddNode(const Node &node) {
  auto result = nodes_.emplace(node.GetId(), node);
  return &(result.first->second);
}

Edge *Graph::AddEdge(const Edge &edge) {
  auto result = edges_.emplace(edge);
  return &(const_cast<Edge &>(*(result.first)));
}

Node *Graph::FindNode(const NodeId &id) {
  auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const Node *Graph::FindNode(const NodeId &id) const {
  return const_cast<Graph &>(*this).FindNode(id);
}

std::pair<const Node *, const Node *> Graph::FindNodePair(const Edge &edge) const {
  return std::make_pair(FindNode(edge.GetSrc().getNodeId()), FindNode(edge.GetDst().getNodeId()));
}

std::pair<Node *, Node *> Graph::FindNodePair(const Edge &edge) {
  return std::make_pair(FindNode(edge.GetSrc().getNodeId()), FindNode(edge.GetDst().getNodeId()));
}

void Graph::Accept(GraphVisitor &visitor) const {
  visitor.Visit(*this);
  std::for_each(nodes_.begin(), nodes_.end(), [&visitor](const auto &node) { visitor.Visit(node.second); });
  std::for_each(edges_.begin(), edges_.end(), [&visitor](const auto &edge) { visitor.Visit(edge); });
}

Status Graph::Layout(const LayoutOption *option) const {
  return GraphLayout::GetInstance().Layout(*this, option);
}

EG_NS_END
