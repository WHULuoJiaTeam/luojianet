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

#ifndef HF37ACE88_F726_4AA3_8599_ED7A888AA623
#define HF37ACE88_F726_4AA3_8599_ED7A888AA623

#include <vector>
#include "easy_graph/graph/node_id.h"
#include "easy_graph/infra/operator.h"
#include "easy_graph/infra/ext_traits.h"
#include "easy_graph/graph/box.h"

EG_NS_BEGIN

struct GraphVisitor;
struct Graph;

struct Node {
  template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, Graph)>
  Node(const NodeId &id, const GRAPHS &... graphs) : id_(id), subgraphs_{&graphs...} {}

  template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, Graph)>
  Node(const NodeId &id, const BoxPtr &box, const GRAPHS &... graphs) : id_(id), box_(box), subgraphs_{&graphs...} {}

  __DECL_COMP(Node);

  NodeId GetId() const;

  Node &Packing(const BoxPtr &);

  template<typename Anything>
  Anything *Unpacking() const {
    if (!box_)
      return nullptr;
    return BoxUnpacking<Anything>(box_);
  }

  Node &AddSubgraph(const Graph &);
  void Accept(GraphVisitor &) const;

 private:
  NodeId id_;
  BoxPtr box_;
  std::vector<const Graph *> subgraphs_;
};

EG_NS_END

#endif
