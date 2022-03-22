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

#include "easy_graph/builder/chain_builder.h"
#include "easy_graph/builder/graph_builder.h"

EG_NS_BEGIN

ChainBuilder::ChainBuilder(GraphBuilder &graphBuilder, EdgeType defaultEdgeType)
    : linker(*this, defaultEdgeType), graph_builder_(graphBuilder) {}

ChainBuilder::LinkBuilder *ChainBuilder::operator->() {
  return &linker;
}

ChainBuilder &ChainBuilder::LinkTo(const Node &node, const Link &link) {
  Node *currentNode = graph_builder_.BuildNode(node);
  if (prev_node_) {
    graph_builder_.BuildEdge(*prev_node_, *currentNode, link);
  }
  prev_node_ = currentNode;
  return *this;
}

const Node *ChainBuilder::FindNode(const NodeId &id) const {
  return graph_builder_->FindNode(id);
}

ChainBuilder::LinkBuilder::LinkBuilder(ChainBuilder &chain, EdgeType defaultEdgeType)
    : chain_(chain), default_edge_type_(defaultEdgeType), from_link_(defaultEdgeType) {}

ChainBuilder &ChainBuilder::LinkBuilder::Node(const NodeObj &node) {
  chain_.LinkTo(node, from_link_);
  from_link_.Reset(default_edge_type_);
  return chain_;
}

ChainBuilder &ChainBuilder::LinkBuilder::startLink(const Link &link) {
  this->from_link_ = link;
  return chain_;
}

ChainBuilder &ChainBuilder::LinkBuilder::Ctrl(const std::string &label) {
  return this->Edge(EdgeType::CTRL, UNDEFINED_PORT_ID, UNDEFINED_PORT_ID, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Data(const std::string &label) {
  return this->Edge(EdgeType::DATA, UNDEFINED_PORT_ID, UNDEFINED_PORT_ID, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Data(PortId srcId, PortId dstId, const std::string &label) {
  return this->Edge(EdgeType::DATA, srcId, dstId, label);
}

ChainBuilder &ChainBuilder::LinkBuilder::Edge(EdgeType type, PortId srcPort, PortId dstPort, const std::string &label) {
  return this->startLink(Link(type, label, srcPort, dstPort));
}

EG_NS_END
