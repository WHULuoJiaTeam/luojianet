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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_REPLACE_MOMENTUM_CAST_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_REPLACE_MOMENTUM_CAST_FUSION_H_

#include <memory>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ReplaceMomentumCastFusion : public PatternProcessPass {
 public:
  explicit ReplaceMomentumCastFusion(bool multigraph = true) : PatternProcessPass("replace_momentum_cast", multigraph) {
    var_ = std::make_shared<Var>();
    acc_ = std::make_shared<Var>();
    lr_ = std::make_shared<Var>();
    grad_ = std::make_shared<Var>();
    mom_ = std::make_shared<Var>();
  }
  ~ReplaceMomentumCastFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr var_;
  VarPtr acc_;
  VarPtr lr_;
  VarPtr grad_;
  VarPtr mom_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_REPLACE_MOMENTUM_CAST_FUSION_H_
