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

#include "easy_graph/graph/endpoint.h"

EG_NS_BEGIN

Endpoint::Endpoint(const NodeId &nodeId, const PortId &portId) : node_id_(nodeId), port_id_(portId) {}

__DEF_EQUALS(Endpoint) {
  return (node_id_ == rhs.node_id_) && (port_id_ == rhs.port_id_);
}

__DEF_COMP(Endpoint) {
  if (node_id_ < rhs.node_id_)
    return true;
  if ((node_id_ == rhs.node_id_) && (port_id_ < rhs.port_id_))
    return true;
  return false;
}

NodeId Endpoint::getNodeId() const {
  return node_id_;
}

PortId Endpoint::getPortId() const {
  return port_id_;
}

EG_NS_END
