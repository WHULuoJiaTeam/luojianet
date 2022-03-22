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

#ifndef H5FED5F58_167D_4536_918A_D5FE8F28DD9C
#define H5FED5F58_167D_4536_918A_D5FE8F28DD9C

#include "easy_graph/graph/graph.h"

EG_NS_BEGIN

struct Link;

struct GraphBuilder {
  GraphBuilder(const std::string &name);

  Node *BuildNode(const Node &);
  Edge *BuildEdge(const Node &src, const Node &dst, const Link &);

  Graph &operator*() {
    return graph_;
  }

  const Graph &operator*() const {
    return graph_;
  }

  Graph *operator->() {
    return &graph_;
  }

  const Graph *operator->() const {
    return &graph_;
  }

 private:
  struct NodeInfo {
    PortId inPortMax{0};
    PortId outPortMax{0};
  };

  NodeInfo *FindNode(const NodeId &);
  const NodeInfo *FindNode(const NodeId &) const;

 private:
  std::map<NodeId, NodeInfo> nodes_;
  Graph graph_;
};

EG_NS_END

#endif
