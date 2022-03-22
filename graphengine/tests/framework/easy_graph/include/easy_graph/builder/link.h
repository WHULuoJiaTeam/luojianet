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

#ifndef H35695B82_E9E5_419D_A6B4_C13FB0842C9F
#define H35695B82_E9E5_419D_A6B4_C13FB0842C9F

#include <string>
#include "easy_graph/graph/edge_type.h"
#include "easy_graph/graph/port_id.h"

EG_NS_BEGIN

struct Link {
  explicit Link(EdgeType type) : type_(type) {
    Reset(type);
  }

  Link(EdgeType type, const std::string &label, PortId srcPortId, PortId dstPortId)
      : type_(type), label_(label), src_port_id_(srcPortId), dst_port_id_(dstPortId) {}

  void Reset(EdgeType type) {
    this->type_ = type;
    this->label_ = "";
    this->src_port_id_ = UNDEFINED_PORT_ID;
    this->dst_port_id_ = UNDEFINED_PORT_ID;
  }

  EdgeType type_;
  std::string label_;
  PortId src_port_id_;
  PortId dst_port_id_;
};

EG_NS_END

#endif
