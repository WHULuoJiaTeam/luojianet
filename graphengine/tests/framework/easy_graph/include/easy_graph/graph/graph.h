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

#ifndef H813EC8C1_3850_4320_8AC0_CE071C89B871
#define H813EC8C1_3850_4320_8AC0_CE071C89B871

#include "easy_graph/graph/node.h"
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "easy_graph/infra/status.h"
#include <string>
#include <set>
#include <map>

EG_NS_BEGIN

struct GraphVisitor;
struct LayoutOption;

struct Graph {
  explicit Graph(const std::string &name);

  std::string GetName() const;

  Node *AddNode(const Node &);
  Edge *AddEdge(const Edge &);

  Node *FindNode(const NodeId &);
  const Node *FindNode(const NodeId &) const;

  std::pair<const Node *, const Node *> FindNodePair(const Edge &) const;
  std::pair<Node *, Node *> FindNodePair(const Edge &);

  void Accept(GraphVisitor &) const;

  Status Layout(const LayoutOption *option = nullptr) const;

 private:
  std::string name_;
  std::map<NodeId, Node> nodes_;
  std::set<Edge> edges_;
};

EG_NS_END

#endif
