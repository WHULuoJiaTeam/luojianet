/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STOPGRAD_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STOPGRAD_ELIMINATE_H_

#include "ir/anf.h"
#include "base/core_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore::opt::irpass {
//
// StopGradientEliminater eliminates redundant stop_gradient nodes.
//
class StopGradientEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &start_node) override {
    // We assume that the start_node is a StopGradient cnode.
    AnfNodePtr node = start_node;
    AnfNodePtr input = nullptr;
    while ((input = GetInputStopGradient(node)) != nullptr) {
      node = input;
    }
    if (node != start_node) {
      return node;
    }
    return nullptr;
  }

 private:
  static inline AnfNodePtr GetInputStopGradient(const AnfNodePtr &node) {
    auto &input = node->cast<CNodePtr>()->inputs().at(1);
    if (IsPrimitiveCNode(input, prim::kPrimStopGradient)) {
      return input;
    }
    return nullptr;
  }
};
}  // namespace mindspore::opt::irpass

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STOPGRAD_ELIMINATE_H_
