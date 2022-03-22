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

#include "easy_graph/builder/graph_builder.h"
#include "easy_graph/graph/endpoint.h"
#include "easy_graph/builder/link.h"
#include "easy_graph/infra/log.h"

EG_NS_BEGIN

namespace {
PortId getPortIdBy(const EdgeType &type, const PortId &specifiedPortId, PortId &reservedPortId) {
  if (type == EdgeType::CTRL)
    return 0;
  if (specifiedPortId == UNDEFINED_PORT_ID)
    return reservedPortId++;
  if (specifiedPortId < reservedPortId)
    return specifiedPortId;
  reservedPortId = specifiedPortId;
  return reservedPortId++;
}
}  // namespace

GraphBuilder::GraphBuilder(const std::string &name) : graph_(name) {}

GraphBuilder::NodeInfo *GraphBuilder::FindNode(const NodeId &id) {
  auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const GraphBuilder::NodeInfo *GraphBuilder::FindNode(const NodeId &id) const {
  return const_cast<GraphBuilder &>(*this).FindNode(id);
}

Node *GraphBuilder::BuildNode(const Node &node) {
  auto it = nodes_.find(node.GetId());
  if (it == nodes_.end()) {
    nodes_.emplace(std::make_pair(node.GetId(), NodeInfo()));
  }
  return graph_.AddNode(node);
}

Edge *GraphBuilder::BuildEdge(const Node &src, const Node &dst, const Link &link) {
  NodeInfo *srcInfo = FindNode(src.GetId());
  NodeInfo *dstInfo = FindNode(dst.GetId());

  if (!srcInfo || !dstInfo) {
    EG_ERR("link edge{%d : %s} error!", link.type_, link.label_.c_str());
    return nullptr;
  }

  PortId srcPortId = getPortIdBy(link.type_, link.src_port_id_, srcInfo->outPortMax);
  PortId dstPortId = getPortIdBy(link.type_, link.dst_port_id_, dstInfo->inPortMax);

  EG_DBG("link edge(%d) from (%s:%d) to (%s:%d)", link.type_, src.GetId().c_str(), srcPortId, dst.GetId().c_str(),
         dstPortId);

  return graph_.AddEdge(
      Edge(link.type_, link.label_, Endpoint(src.GetId(), srcPortId), Endpoint(dst.GetId(), dstPortId)));
}

EG_NS_END
