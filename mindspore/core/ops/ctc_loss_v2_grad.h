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

#ifndef MINDSPORE_CORE_OPS_CTC_LOSS_V2_GRAD_H_
#define MINDSPORE_CORE_OPS_CTC_LOSS_V2_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCTCLossV2Grad = "CTCLossV2Grad";
class MIND_API CTCLossV2Grad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CTCLossV2Grad);
  CTCLossV2Grad() : BaseOperator(kNameCTCLossV2Grad) {
    InitIOName(
      {"grad_out", "log_probs", "targets", "input_lengths", "target_lengths", "neg_log_likelihood", "log_alpha"},
      {"grad"});
  }
  void Init() const {}
};

abstract::AbstractBasePtr CTCLossV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimCTCLossV2Ptr = std::shared_ptr<CTCLossV2Grad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CTC_LOSS_V2_GRAD_H_
