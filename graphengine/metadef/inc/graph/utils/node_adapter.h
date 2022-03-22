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

#ifndef INC_GRAPH_UTILS_NODE_ADAPTER_H_
#define INC_GRAPH_UTILS_NODE_ADAPTER_H_

#include "graph/gnode.h"
#include "graph/node.h"

namespace ge {
using NodePtr = std::shared_ptr<Node>;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodeAdapter {
 public:
  static GNode Node2GNode(const NodePtr &node);
  static NodePtr GNode2Node(const GNode &node);
  static GNodePtr Node2GNodePtr(const NodePtr &node);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_NODE_ADAPTER_H_
