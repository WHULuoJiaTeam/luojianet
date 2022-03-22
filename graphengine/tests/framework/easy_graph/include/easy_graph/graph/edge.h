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

#ifndef HE62943BF_1F7C_4EF9_B306_D9C94634EA74
#define HE62943BF_1F7C_4EF9_B306_D9C94634EA74

#include <string>
#include "easy_graph/graph/edge_type.h"
#include "easy_graph/graph/endpoint.h"

EG_NS_BEGIN

struct Edge {
  Edge(const EdgeType type, const std::string &label, const Endpoint &src, const Endpoint &dst);

  __DECL_COMP(Edge);

  EdgeType GetType() const;
  std::string GetLabel() const;

  Endpoint GetSrc() const;
  Endpoint GetDst() const;

 private:
  std::string label_;
  EdgeType type_;
  Endpoint src_;
  Endpoint dst_;
};

EG_NS_END

#endif
