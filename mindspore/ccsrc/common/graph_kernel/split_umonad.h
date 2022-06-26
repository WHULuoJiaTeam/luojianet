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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_UMONAD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_UMONAD_H_

#include "backend/common/optimizer/optimizer.h"
#include "common/graph_kernel/adapter/graph_kernel_expander_with_py.h"
namespace mindspore::graphkernel {
class SplitAssign : public opt::PatternProcessPass {
 public:
  explicit SplitAssign(bool multigraph = true) : PatternProcessPass("split_assign", multigraph) {}
  ~SplitAssign() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class OpUMonadExpander : public DefaultExpander {
 public:
  explicit OpUMonadExpander(size_t input_idx) : input_idx_(input_idx) {}
  virtual ~OpUMonadExpander() = default;
  AnfNodePtr Run(const AnfNodePtr &node) override;

 private:
  size_t input_idx_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_UMONAD_H_
