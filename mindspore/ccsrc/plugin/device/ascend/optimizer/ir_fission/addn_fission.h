/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADDN_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADDN_FISSION_H_

#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
constexpr size_t kAddnInputsDivisor = 63;
class AddnFission : public PatternProcessPass {
 public:
  explicit AddnFission(bool multigraph = true)
      : PatternProcessPass("addn_fission", multigraph), inputs_divisor_(kAddnInputsDivisor) {}
  ~AddnFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateNewAddn(const FuncGraphPtr &func_graph, const CNodePtr &origin_addn_cnode, size_t begin_index,
                           size_t offset) const;
  size_t inputs_divisor_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_ADDN_FISSION_H_
