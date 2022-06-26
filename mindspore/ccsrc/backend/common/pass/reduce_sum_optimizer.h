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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REDUCE_SUM_OPTIMIZER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REDUCE_SUM_OPTIMIZER_H_
#include <memory>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ReduceSumOptimizer : public PatternProcessPass {
 public:
  explicit ReduceSumOptimizer(bool multigraph = true) : PatternProcessPass("reduce_sum_optimizer", multigraph) {}
  ~ReduceSumOptimizer() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  AnfNodePtr NewRankOp(const AnfNodePtr &cnode, const KernelGraphPtr &kernel_graph) const;
  AnfNodePtr NewRangeOp(const AnfNodePtr &rank_op, const KernelGraphPtr &kernel_graph) const;
  AnfNodePtr InsertAssistNode(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph) const;
  AnfNodePtr NewAssistValueNode(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REDUCE_SUM_OPTIMIZER_H_
