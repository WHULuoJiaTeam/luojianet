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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADD_RELU_GRAD_V2_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADD_RELU_GRAD_V2_FUSION_H_

#include <memory>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class AddReluGradV2Fusion : public PatternProcessPass {
 public:
  explicit AddReluGradV2Fusion(bool multigraph = true) : PatternProcessPass("add_relu_grad", multigraph) {
    x1_ = std::make_shared<Var>();
    x2_ = std::make_shared<Var>();
    mask_ = std::make_shared<Var>();
  }
  ~AddReluGradV2Fusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr x1_;
  VarPtr x2_;
  VarPtr mask_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADD_RELUGRAD_FUSION_H_
