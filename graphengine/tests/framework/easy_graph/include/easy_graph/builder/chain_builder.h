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

#ifndef HDF50E564_F050_476A_A479_F82B20F35C84
#define HDF50E564_F050_476A_A479_F82B20F35C84

#include "easy_graph/builder/link.h"
#include "easy_graph/graph/node_id.h"
#include "easy_graph/graph/node.h"

EG_NS_BEGIN

struct GraphBuilder;
struct Graph;
struct Edge;

struct ChainBuilder {
  ChainBuilder(GraphBuilder &graphBuilder, EdgeType defaultEdgeType);

  struct LinkBuilder {
    using NodeObj = ::EG_NS::Node;
    using EdgeObj = ::EG_NS::Edge;

    LinkBuilder(ChainBuilder &chain, EdgeType defaultEdgeType);

    ChainBuilder &Node(const NodeObj &node);

    template<typename... PARAMS>
    ChainBuilder &Node(const NodeId &id, const PARAMS &... params) {
      auto node = chain_.FindNode(id);
      if (node) {
        return this->Node(*node);
      }
      return this->Node(NodeObj(id, params...));
    }

    ChainBuilder &Ctrl(const std::string &label = "");
    ChainBuilder &Data(const std::string &label = "");

    ChainBuilder &Data(PortId srcId = UNDEFINED_PORT_ID, PortId dstId = UNDEFINED_PORT_ID,
                       const std::string &label = "");

    ChainBuilder &Edge(EdgeType type, PortId srcId = UNDEFINED_PORT_ID, PortId dstId = UNDEFINED_PORT_ID,
                       const std::string &label = "");

   private:
    ChainBuilder &startLink(const Link &);

   private:
    ChainBuilder &chain_;
    EdgeType default_edge_type_;
    Link from_link_;
  } linker;

  LinkBuilder *operator->();

 private:
  ChainBuilder &LinkTo(const Node &, const Link &);
  const Node *FindNode(const NodeId &) const;

 private:
  Node *prev_node_{nullptr};
  GraphBuilder &graph_builder_;
};

EG_NS_END

#endif
