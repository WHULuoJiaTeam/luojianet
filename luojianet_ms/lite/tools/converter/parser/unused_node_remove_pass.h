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
#ifndef LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_NODE_REMOVE_PASS_H_
#define LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_NODE_REMOVE_PASS_H_

#include <set>
#include <string>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"
#include "luojianet_ms/lite/include/errorcode.h"

using luojianet_ms::lite::STATUS;
namespace luojianet_ms::opt {
class UnusedNodeRemovePass : public Pass {
 public:
  UnusedNodeRemovePass() : Pass("remove_unused_node_pass") {}
  ~UnusedNodeRemovePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS ProcessGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *has_visited);
};
}  // namespace luojianet_ms::opt

#endif  // LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_GRAPH_UNUSED_NODE_REMOVE_PASS_H_
