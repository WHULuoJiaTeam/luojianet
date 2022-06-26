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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONFUSION_MUL_GRAD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONFUSION_MUL_GRAD_FUSION_H_

#include <memory>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ConfusionMulGradFusion : public PatternProcessPass {
 public:
  explicit ConfusionMulGradFusion(bool multigraph = true)
      : PatternProcessPass("confusion_mul_grad_fusion", multigraph) {
    input2_ = std::make_shared<Var>();
    input3_ = std::make_shared<Var>();
  }
  ~ConfusionMulGradFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateFusionNode(const FuncGraphPtr &graph, const CNodePtr &reduce_sum, const AnfNodePtr &mul0_anf,
                            const AnfNodePtr &input3) const;
  VarPtr input2_;
  VarPtr input3_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONFUSION_MUL_GRAD_FUSION_H_
