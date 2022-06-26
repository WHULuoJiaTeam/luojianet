/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
#include <memory>
#include <vector>
#include <string>
#include "backend/common/optimizer/pass.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
class Expander {
 public:
  virtual AnfNodePtr Run(const AnfNodePtr &node) = 0;
  virtual ~Expander() = default;
};
using ExpanderPtr = std::shared_ptr<Expander>;

class DefaultExpander : public Expander {
 public:
  AnfNodePtr Run(const AnfNodePtr &node) override;
  virtual ~DefaultExpander() = default;

 protected:
  virtual AnfNodePtr CreateExpandGraphKernel(const FuncGraphPtr &new_func_graph, const CNodePtr &old_node);
  virtual FuncGraphPtr CreateExpandFuncGraph(const CNodePtr &node);
};

class GraphKernelExpander : public opt::Pass {
 public:
  GraphKernelExpander() : Pass("graph_kernel_expander") {}
  explicit GraphKernelExpander(const std::string &name) : Pass(name) {}
  ~GraphKernelExpander() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  virtual ExpanderPtr GetExpander(const AnfNodePtr &node);
  virtual std::vector<PrimitivePtr> InitOpList() = 0;
  virtual bool DoExpand(const FuncGraphPtr &func_graph);
  virtual bool CanExpand(const CNodePtr &node) const {
    return std::any_of(expand_ops_.begin(), expand_ops_.end(),
                       [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  }

 private:
  std::vector<PrimitivePtr> expand_ops_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
