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

#ifndef MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
#define MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSmoothL1LossGrad = "SmoothL1LossGrad";
class MIND_API SmoothL1LossGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SmoothL1LossGrad);
  SmoothL1LossGrad() : BaseOperator(kNameSmoothL1LossGrad) {}
  void Init();
  void Init(const float beta);
  void set_beta(const float beta);
  float get_beta() const;
};
abstract::AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimSmoothL1LossGradPtr = std::shared_ptr<SmoothL1LossGrad>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
