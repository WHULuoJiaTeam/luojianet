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

#include "easy_graph/graph/edge.h"

EG_NS_BEGIN

Edge::Edge(const EdgeType type, const std::string &label, const Endpoint &src, const Endpoint &dst)
    : type_(type), label_(label), src_(src), dst_(dst) {}

__DEF_EQUALS(Edge) {
  return (type_ == rhs.type_) && (src_ == rhs.src_) && (dst_ == rhs.dst_);
}

__DEF_COMP(Edge) {
  if (src_ < rhs.src_)
    return true;
  if ((src_ == rhs.src_) && (dst_ < rhs.dst_))
    return true;
  if ((src_ == rhs.src_) && (dst_ < rhs.dst_) && (type_ < rhs.type_))
    return true;
  return false;
}

EdgeType Edge::GetType() const {
  return type_;
}

std::string Edge::GetLabel() const {
  return label_;
}

Endpoint Edge::GetSrc() const {
  return src_;
}
Endpoint Edge::GetDst() const {
  return dst_;
}

EG_NS_END
