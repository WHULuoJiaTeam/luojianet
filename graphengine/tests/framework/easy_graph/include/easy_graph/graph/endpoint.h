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

#ifndef H8DB48A37_3257_4E15_8869_09E58221ADE8
#define H8DB48A37_3257_4E15_8869_09E58221ADE8

#include "easy_graph/graph/node_id.h"
#include "easy_graph/graph/port_id.h"
#include "easy_graph/infra/operator.h"

EG_NS_BEGIN

struct Endpoint {
  Endpoint(const NodeId &, const PortId &);

  __DECL_COMP(Endpoint);

  NodeId getNodeId() const;
  PortId getPortId() const;

 private:
  NodeId node_id_;
  PortId port_id_;
};

EG_NS_END

#endif
